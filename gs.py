#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import logging
import pickle
import shutil
import uuid
from random import randint

import lpips
import matplotlib.pyplot as plt
from fused_ssim import fused_ssim as ssim

from arguments import *
from arguments import args_init
from metrics import evaluate as evaluate_metrics
from metrics_half import evaluate as evaluate_metrics_half
from render import *
from utils import network_gui
from utils.image_utils import psnr

torch.set_float32_matmul_precision("high")

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def train(args):
    device = torch.device(args.render_device)
    dataset_name = args.scene_name
    log_file_path = os.path.join(args.model_path, "logs",
                                 f"({time.strftime('%Y-%m-%d_%H-%M-%S')})_iteration({args.iterations})_({dataset_name}).log")
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    logging.basicConfig(filename=log_file_path, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info(f"Experiment Configuration: {args}")
    logging.info(f"Model initialization and Data reading ...")
    # save args
    with open(os.path.join(args.model_path, 'cfg_arg.pkl'), 'wb') as file:
        pickle.dump(args, file)
    first_iter = 0

    tb_writer = prepare_output_and_logger(args)
    if args.depth_is_inverted:
        from models.gs.gaussian_model_inv_depth import GaussianModelInvDepth
        gaussians = GaussianModelInvDepth(args)
    else:
        gaussians = GaussianModel(args)
    scene = Scene(args, gaussians, shuffle=False)
    gaussians.training_setup(args)

    if args.deblur:
        blur_blend_embedding = torch.nn.Embedding(
            len(scene.train_cameras), args.blur_sample_num, device=device)
        blur_blend_embedding.weight = torch.nn.Parameter(torch.ones(
            len(scene.train_cameras), args.blur_sample_num, device=device))
        optimizer = torch.optim.Adam([
            {'params': blur_blend_embedding.parameters(),
             'lr': 1e-3, "name": "blur blend parameters"},
        ], lr=0.0, eps=1e-15)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=(1e-6 / 1e-3) ** (1. / args.iterations))
    else:
        args.blur_sample_num = 1

    render_temp_path = os.path.join(args.model_path, "train_temp_rendering")
    gt_temp_path = os.path.join(args.model_path, "train_temp_gt")
    if os.path.exists(render_temp_path):
        shutil.rmtree(render_temp_path)
    if os.path.exists(gt_temp_path):
        shutil.rmtree(gt_temp_path)
    os.makedirs(render_temp_path, exist_ok=True)
    os.makedirs(gt_temp_path, exist_ok=True)

    if args.use_features_mask:
        render_temp_mask_path = os.path.join(args.model_path, "train_mask_temp_rendering")
        if os.path.exists(render_temp_mask_path):
            shutil.rmtree(render_temp_mask_path)
        os.makedirs(render_temp_mask_path, exist_ok=True)

    bg_color = [1, 1, 1] if args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float, device=device)

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_depth_loss_for_log = 0.0
    ema_psnr_for_log = 0.0
    progress_bar = tqdm(range(first_iter, args.iterations), desc="Training")
    first_iter += 1
    n_cameras = len(scene.train_cameras)
    record_loss = []
    ech_loss = 0
    if args.use_lpips_loss:  # vgg alex
        lpips_criteria = lpips.LPIPS(net='vgg').to(device)
        if args.compile:
            lpips_criteria = torch.compile(lpips_criteria)
    logging.info(f"Start training ...")

    if args.warm_up_iter > 0:
        gaussians.set_learning_rate("box_coord", 0.0)

    depth_l1_weight = get_expon_lr_func(args.depth_l1_weight_init, args.depth_l1_weight_final, max_steps=args.iterations)

    for iteration in range(first_iter, args.iterations + 1):
        if network_gui.conn is None:
            network_gui.try_connect()
        while network_gui.conn is not None:
            try:
                net_image_bytes = None
                custom_cam, do_training, args.convert_SHs_python, args.compute_cov3D_python, keep_alive, scaling_modifier = network_gui.receive()
                if custom_cam is not None:
                    net_image = gaussians.render(custom_cam, args, background, scaling_modifier)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, args.source_path)
                if do_training and ((iteration < int(args.iterations)) or not keep_alive):
                    break
            except Exception:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration, args.warm_up_iter)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.train_cameras.copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        # plt.figure(dpi=200)
        # plt.subplot(2, 1, 1)
        # plt.imshow(viewpoint_cam.original_image.permute(1, 2, 0).cpu().numpy())
        # plt.title("Original Image")
        # plt.subplot(2, 1, 2)
        # plt.imshow(viewpoint_cam.depth.cpu().numpy())
        # plt.title("Depth Image")
        # plt.tight_layout()
        # os.makedirs(f"{args.model_path}/vis", exist_ok=True)
        # plt.savefig(f"{args.model_path}/vis/image_{iteration}.jpg")

        if args.deblur:
            blur_weight = blur_blend_embedding(
                torch.tensor(viewpoint_cam.uid, device=device))
            blur_weight /= torch.sum(blur_weight)
        else:
            blur_weight = 1.0

        # Render
        if (iteration - 1) == args.debug_from:
            args.debug = True

        bg = torch.rand(3, device=device) if args.random_background else background

        image = 0
        depth = 0
        radii = torch.zeros(len(gaussians), dtype=torch.float, device=device)
        viewspace_point_tensors = []
        viewspace_point_tensor_data = 0
        visibility_filter = torch.zeros(len(gaussians), dtype=torch.bool, device=device)

        gt_image = viewpoint_cam.original_image.to(device)

        if not args.non_uniform:
            blur_weight = 1.0 / args.blur_sample_num

        for idx in range(args.blur_sample_num):
            alpha = idx / (max(1, args.blur_sample_num - 1))
            render_pkg = gaussians.render(viewpoint_cam, args, bg, interp_alpha=alpha)
            image_, viewspace_point_tensor_, visibility_filter_, radii_, depth_ = (
                render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"],
                render_pkg["radii"], render_pkg["depth"])
            image += image_ * blur_weight
            if depth_ is not None and args.use_depth_loss:
                depth += depth_ * blur_weight
            radii = torch.max(radii_, radii)
            visibility_filter |= visibility_filter_
            viewspace_point_tensors.append(viewspace_point_tensor_)
            viewspace_point_tensor_data += viewspace_point_tensor_ * blur_weight

        mask = None
        if args.use_features_mask and iteration > args.features_mask_iters:  # 2500
            mask = F.interpolate(gaussians.features_mask, size=image.shape[-2:])[0]
        elif args.use_masks and viewpoint_cam.mask is not None:
            mask = viewpoint_cam.mask.to(device)
        if mask is not None:
            image *= mask
            gt_image = gt_image * mask
        Ll1 = F.l1_loss(image, gt_image)
        loss = (1.0 - args.lambda_dssim) * Ll1 + args.lambda_dssim * (1.0 - ssim(image[None], gt_image[None]))

        if args.use_features_mask and iteration > args.features_mask_iters:  # 2500
            loss += (torch.square(1 - gaussians.features_mask)).mean() * args.features_mask_loss_coef

        if args.use_scaling_loss:
            loss += torch.abs(gaussians.get_scaling).mean() * args.scaling_loss_coef

        if args.use_lpips_loss:
            loss += lpips_criteria(image, gt_image).mean() * args.lpips_loss_coef

        if (gaussians.use_kmap_pjmap or gaussians.use_okmap) and args.use_box_coord_loss:
            loss += torch.relu(torch.abs(gaussians.map_pts_norm) - 1).mean() * args.box_coord_loss_coef

        if args.use_depth_loss and viewpoint_cam.depth is not None:
            gt_depth = viewpoint_cam.depth.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()

            depth_loss_pure = torch.abs((gt_depth - depth) * depth_mask).mean()
            depth_loss = depth_l1_weight(iteration) * depth_loss_pure
            loss += depth_loss

        psnr_ = psnr(image, gt_image).double().mean()

        if loss.isnan():
            gaussians.set_eval(True)
            training_report(tb_writer, iteration, Ll1, loss, iter_start.elapsed_time(iter_end),
                            args.test_iterations, scene, gaussians.render, (args, background))
            gaussians.set_eval(False)
            logging.info(f"[ITER {iteration}] Saving Gaussians")
            print(f"\n[ITER {iteration}] Saving Gaussians")
            scene.save(iteration)
            raise ValueError(f"Loss is NaN at iteration {iteration}")

        loss.backward()

        viewspace_point_tensor = viewspace_point_tensor_data.clone().detach().requires_grad_(True)
        viewspace_point_tensor.grad = None
        for viewspace_point_tensor_ in viewspace_point_tensors:
            if viewspace_point_tensor.grad is None:
                viewspace_point_tensor.grad = viewspace_point_tensor_.grad
            else:
                viewspace_point_tensor.grad = torch.max(
                    viewspace_point_tensor.grad, viewspace_point_tensor_.grad)

        iter_end.record()
        ech_loss += loss.item()
        if iteration % n_cameras == 0:
            logging.info(f'Iteration {iteration}/{args.iterations}, Loss: {ech_loss / n_cameras}')
            logging.info(f"Iteration {iteration}: # Gaussian Points: {len(gaussians)}")
            record_loss.append(ech_loss / n_cameras)
            ech_loss = 0

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if args.use_depth_loss:
                ema_depth_loss_for_log = 0.4 * depth_loss.item() + 0.6 * ema_depth_loss_for_log
            ema_psnr_for_log = 0.4 * psnr_ + 0.6 * ema_psnr_for_log
            total_point = len(gaussians)
            if iteration % 100 == 0 or iteration == 1:
                image = image.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
                image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
                cv.imwrite(os.path.join(render_temp_path,
                                        f"iter{iteration:05d}_{viewpoint_cam.image_name.replace('.png', '.jpg')}"),
                           image,
                           [int(cv.IMWRITE_JPEG_QUALITY), 100])
                gt_image = gt_image.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
                gt_image = cv.cvtColor(gt_image, cv.COLOR_RGB2BGR)
                cv.imwrite(os.path.join(gt_temp_path,
                                        f"iter{iteration:05d}_{viewpoint_cam.image_name.replace('.png', '.jpg')}"),
                           gt_image,
                           [int(cv.IMWRITE_JPEG_QUALITY), 100])
                # torchvision.utils.save_image(image, os.path.join(render_temp_path, f"iter{iteration}_"+viewpoint_cam.image_name + ".png"))
                # torchvision.utils.save_image(gt_image, os.path.join(gt_temp_path, f"iter{iteration}_"+viewpoint_cam.image_name + ".png"))
                if args.use_features_mask:
                    torchvision.utils.save_image(gaussians.features_mask.repeat(1, 3, 1, 1),
                                                 os.path.join(render_temp_mask_path,
                                                              f"iter{iteration:05d}_{viewpoint_cam.image_name.replace('.jpg', '.png')}"))
            d = {"Loss": f"{ema_loss_for_log:.7f}",
                 "PSNR": f"{psnr_:.2f}",
                 "#": f"{total_point}"}
            d = {**d, "DepthLoss": f"{ema_depth_loss_for_log:.7f}"} if args.use_depth_loss else d
            progress_bar.set_postfix(d)
            progress_bar.update()
            if iteration == args.iterations:
                progress_bar.close()

            # Log and save
            gaussians.set_eval(True)
            training_report(tb_writer, iteration, Ll1, loss, iter_start.elapsed_time(iter_end),
                            args.test_iterations, scene, gaussians.render, (args, background))
            gaussians.set_eval(False)
            if iteration in args.save_iterations:
                logging.info(f"[ITER {iteration}] Saving Gaussians")
                print(f"\n[ITER {iteration}] Saving Gaussians")
                scene.save(iteration)

            # Densification
            if iteration < args.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                                     radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter, image.shape[2],
                                                  image.shape[1])

                if iteration > args.densify_from_iter and iteration % args.densification_interval == 0:
                    size_threshold = 20 if iteration > args.opacity_reset_interval else None
                    gaussians.densify_and_prune(args.densify_grad_threshold, args.opacity_threshold,
                                                scene.cameras_extent, size_threshold, args.prune_more)

                if iteration % args.opacity_reset_interval == 0 or (
                        args.white_background and iteration == args.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < args.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

                if args.deblur:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    viewpoint_cam.update(iteration)

    # drawing training loss curve
    fig = plt.figure(dpi=200)
    logging.info("Drawing Training loss curve")
    print("\nDrawing Training loss curve")
    plt.plot(record_loss)
    plt.xlabel('Epochs')
    plt.ylabel('Training Loss')
    plt.title('Training Error Curve')
    os.makedirs(os.path.join(scene.model_path, "train", f"ours_{iteration}"), exist_ok=True)
    plt.tight_layout()
    fig.savefig(os.path.join(scene.model_path, "train", f"ours_{iteration}", "training_loss.jpg"))
    # render result and evaluate metrics
    with torch.no_grad():
        if args.render_after_train:
            gaussians.set_eval(True)
            torch.cuda.empty_cache()
            if args.scene_name != "lego":
                logging.info(f"Rendering testing set [{len(scene.test_cameras)}] ...")
                render_set(args.model_path, "test", iteration, scene.test_cameras, gaussians, args,
                           background, render_multi_view=False, render_s2d_inter=False)

                torch.cuda.empty_cache()
                logging.info(f"Rendering training set [{len(scene.train_cameras)}] ...")
                render_set(args.model_path, "train", iteration, scene.train_cameras, gaussians, args, background)

                if gaussians.color_net_type in ["naive"]:
                    logging.info(f"Rendering training set's intrinsic image [{len(scene.train_cameras)}] ...")

                    render_intrinsics(args.model_path, "train", iteration, scene.train_cameras, gaussians, args,
                                      background)

                logging.info(f"Test rendering speed [{len(scene.train_cameras)}] ...")
                avg_rendering_speed = test_rendering_speed(scene.train_cameras, gaussians, args, background)
                logging.info(f"rendering speed: {avg_rendering_speed}s/image")
                if gaussians.color_net_type == "naive":
                    logging.info(f"Test rendering speed using cache [{len(scene.train_cameras)}] ...")
                    avg_rendering_speed = test_rendering_speed(scene.train_cameras, gaussians, args, background,
                                                               use_cache=True)
                    logging.info(f"rendering speed using cache: {avg_rendering_speed}s/image")

            else:
                logging.info(f"Rendering testing set [{len(scene.test_cameras)}] ...")
                render_lego(args.model_path, "test", iteration, scene.test_cameras, scene.train_cameras[0],
                            gaussians, args, background)

            render_multiview_video(args, "train", scene.train_cameras, gaussians, background)

            # appearance tuning
            render_interpolation(args, "train", args.iterations, scene.train_cameras, gaussians, background)

            if args.metrics_after_train:
                logging.info("Evaluating metrics on testing set ...")
                evaluate_metrics([args.model_path], use_logs=True)
                logging.info("Evaluating metrics half image on testing set ...")
                evaluate_metrics_half([args.model_path], use_logs=True)
            gaussians.set_eval(False)


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print(f"Output folder: {args.model_path}")
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, Ll1, loss, elapsed, test_iterations, scene: Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in test_iterations and scene.scene_name != "lego":
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras': scene.test_cameras},
                              {'name': 'train',
                               'cameras': [scene.train_cameras[idx % len(scene.train_cameras)] for idx in
                                           range(5, 500, 25)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = renderFunc(viewpoint, *renderArgs)["render"]
                    gt_image = viewpoint.original_image.to(image.device)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(f"{config['name']}_view_{viewpoint.image_name}/render",
                                             image[None], global_step=iteration)
                        if iteration == test_iterations[0]:
                            tb_writer.add_images(f"{config['name']}_view_{viewpoint.image_name}/ground_truth",
                                                 gt_image[None], global_step=iteration)
                    l1_test += F.l1_loss(image, gt_image).double().mean()
                    psnr_test += psnr(image, gt_image).double().mean()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print(f"\n[ITER {iteration}] Evaluating {config['name']}: L1 {l1_test}, PSNR {psnr_test}")
                logging.info(
                    f"[ITER {iteration}] Evaluating {config['name']}: L1 {l1_test}, PSNR {psnr_test}")
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', len(scene.gaussians), iteration)
        torch.cuda.empty_cache()

    elif iteration in test_iterations and scene.scene_name == "lego":
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras': scene.test_cameras},)
        assert scene.gaussians.color_net_type in ["naive"], "color_net_type should be naive"
        _ = scene.gaussians.render(scene.train_cameras[0], pipe=renderArgs[0], bg_color=renderArgs[1],
                                   store_cache=True)["render"]
        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = renderFunc(viewpoint, pipe=renderArgs[0], bg_color=renderArgs[1], use_cache=True)["render"]
                    gt_image = viewpoint.original_image.to(image.device)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(f"{config['name']}_view_{viewpoint.image_name}/render",
                                             image[None], global_step=iteration)
                        if iteration == test_iterations[0]:
                            tb_writer.add_images(f"{config['name']}_view_{viewpoint.image_name}/ground_truth",
                                                 gt_image[None], global_step=iteration)
                    l1_test += F.l1_loss(image, gt_image).double().mean()
                    psnr_test += psnr(image, gt_image).double().mean()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print(f"\n[ITER {iteration}] Evaluating {config['name']}: L1 {l1_test}, PSNR {psnr_test}")
                logging.info(
                    f"[ITER {iteration}] Evaluating {config['name']}: L1 {l1_test}, PSNR {psnr_test}")


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[i * 5000 for i in range(1, 20)])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000])

    parser.add_argument("--render_after_train", action='store_true', default=True)
    parser.add_argument("--metrics_after_train", action='store_true', default=True)
    parser.add_argument("--data_perturb", nargs="+", type=str, default=[])  # for lego ["color","occ"]

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    args = args_init.argument_init(args)

    # Initialize system state (RNG)
    fix_seed()

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    args.position_lr_max_steps = args.iterations

    train(args)

    # All done
    print("\nTraining complete.")
