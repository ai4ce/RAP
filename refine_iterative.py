import json
import os

import numpy as np
import torch
import wandb
from fused_ssim import fused_ssim as ssim
from matplotlib import pyplot as plt
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from arguments import ModelParams, OptimizationParams, get_combined_args
from arguments.options import config_parser
from dataset_loaders.cambridge_scenes import Cambridge
from dataset_loaders.colmap_dataset import ColmapDataset
from dataset_loaders.seven_scenes import SevenScenes
from models.apr.rapnet import RAPNet as RAPNet
from models.gs.gaussian_model import GaussianModel
from utils.cameras import LearnableCamera, CamParams
from utils.eval_utils import get_pose_error, vis_pose
from utils.general_utils import search_for_max_iteration, fix_seed, safe_path
from utils.model_utils import freeze_bn_layer

torch.set_float32_matmul_precision('high')


def refine(args, run):
    if args.dataset_type == '7Scenes':
        dataset_class = SevenScenes
    elif args.dataset_type == 'Colmap':
        dataset_class = ColmapDataset
    elif args.dataset_type == 'Cambridge':
        dataset_class = Cambridge
    else:
        raise ValueError(f"Unknown dataset type: {args.dataset_type}")

    with open(f"{args.model_path}/cameras.json") as f:
        camera = json.load(f)[0]
    rap_cam_params = CamParams(camera, args.rap_resolution, args.device)
    rap_hw = (rap_cam_params.h, rap_cam_params.w)
    gs_cam_params = CamParams(camera, args.resolution, args.device)
    h, w = (gs_cam_params.h, gs_cam_params.w)

    kwargs = dict(data_path=args.datadir, hw=rap_hw, hw_gs=(h, w))
    val_set = dataset_class(train=False, test_skip=args.test_skip, **kwargs)
    test_dl = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=1)

    model = RAPNet(args=args).to(args.device).eval()
    if args.pretrained_model_path and os.path.exists(args.pretrained_model_path):
        print("Loading RAPNet from", args.pretrained_model_path)
        model.load_state_dict(torch.load(args.pretrained_model_path, map_location=args.device, mmap=True))
    else:
        print("Warning: No checkpoint found at", args.pretrained_model_path)
    if args.freeze_batch_norm:
        model = freeze_bn_layer(model)
    if args.compile_model:
        model = torch.compile(model, mode="reduce-overhead")

    gaussians = GaussianModel(args)
    i = search_for_max_iteration(os.path.join(args.model_path, "ckpts_point_cloud"))
    gaussians.load_ckpt_ply(os.path.join(args.model_path, "ckpts_point_cloud", f"iteration_{i}", "point_cloud.ply"))
    bg_color = [1, 1, 1] if args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float, device=args.device)
    gaussians.set_eval(True)
    args.deblur = False

    errors_trans = []
    errors_rot = []
    pred_trans = []

    vis_root = f"vis/{args.run_name}"

    for i, (data, pose, img_orig, img_name) in enumerate(test_dl):
        data = data[0].to(args.device)  # input
        pose = pose.reshape(3, 4)  # label
        img_orig = img_orig[0].to(args.device)
        img_name = img_name[0]
        tmp = np.eye(4)
        tmp[:3, :] = pose.numpy()
        pose_inv = np.linalg.inv(tmp)
        camera = LearnableCamera(None, img_orig, None, pose_inv, pose_inv, gs_cam_params.fx, gs_cam_params.fy)
        image_rendered = gaussians.render(camera, args, background)

        with torch.no_grad():
            _, pred_pose = model(data)

        tmp = np.eye(4)
        tmp[:3] = pred_pose.reshape(3, 4).to("cpu", torch.double).numpy()
        predicted_pose_inv = np.linalg.inv(tmp)
        if gaussians is not None:
            camera = LearnableCamera(None, img_orig, None, pose_inv, predicted_pose_inv, gs_cam_params.fx, gs_cam_params.fy)
            image = gaussians.render(camera, args, background)
            image_before = image
            with tqdm(range(100), desc=f"Refining {i}/{len(test_dl)}") as pbar:
                for _ in pbar:
                    gt_image = camera.original_image
                    Ll1 = F.l1_loss(image, gt_image)
                    loss = (1.0 - args.lambda_dssim) * Ll1 + args.lambda_dssim * (1.0 - ssim(image, gt_image))
                    loss.backward()
                    camera.update()
                    image = gaussians.render(camera, args, background)
                    pbar.set_postfix_str(f"Loss: {loss.item()}")

            with torch.no_grad():
                plt.close("all")
                plt.figure(figsize=(8, 6), dpi=200)
                plt.subplot(2, 2, 1)
                plt.imshow(camera.original_image.permute(1, 2, 0).cpu().numpy())
                plt.title("GT Image")
                plt.axis("off")
                plt.subplot(2, 2, 2)
                plt.imshow(image_rendered.permute(1, 2, 0).cpu().numpy())
                plt.title("GT Rendered Image")
                plt.axis("off")
                plt.subplot(2, 2, 3)
                plt.imshow(image_before.permute(1, 2, 0).cpu().numpy())
                plt.title("Test Rendered Image")
                plt.axis("off")
                plt.subplot(2, 2, 4)
                plt.imshow(image.permute(1, 2, 0).cpu().numpy())
                plt.title("Refined Image")
                plt.axis("off")
                plt.tight_layout()
                plt.savefig(safe_path(f"{vis_root}/{img_name}.jpg"), pil_kwargs={"quality": 100})

                pred_pose = camera.world_view_transform.detach().T.inverse().cpu()

        error_trans, error_rot = get_pose_error(pose, pred_pose)
        error_trans = error_trans.cpu().numpy()
        error_rot = error_rot.cpu().numpy()
        errors_trans.append(error_trans)
        errors_rot.append(error_rot)
        # print ('Iteration: {} Error XYZ (m): {} Error Q (degrees): {}'.format(i, error_x, theta))

        # save results for visualization
        pred_trans.append(pred_pose[:, :3, 3].cpu().numpy())

    errors_trans = np.hstack(errors_trans)
    errors_rot = np.hstack(errors_rot)
    pred_trans = np.vstack(pred_trans)
    gt_trans = val_set.poses.reshape((-1, 3, 4))[:, :3, 3]

    success_condition_5 = (errors_trans < 0.05) & (errors_rot < 5)
    success_condition_2 = (errors_trans < 0.02) & (errors_rot < 2)
    successful_count_5 = np.sum(success_condition_5)
    successful_count_2 = np.sum(success_condition_2)
    print(f"Successful count (5cm/5degree): {successful_count_5}")
    print(f"Successful count (2cm/2degree): {successful_count_2}")

    success_rate_5 = successful_count_5 / errors_trans.shape[0]
    success_rate_2 = successful_count_2 / errors_trans.shape[0]
    print(f"Success rate (5cm/5degree): {success_rate_5:.5%}")
    print(f"Success rate (2cm/2degree): {success_rate_2:.5%}")

    median_trans, median_rot = np.median(errors_trans), np.median(errors_rot)
    mean_trans, mean_rot = np.mean(errors_trans), np.mean(errors_rot)
    max_trans, max_rot = np.max(errors_trans), np.max(errors_rot)
    min_trans, min_rot = np.min(errors_trans), np.min(errors_rot)
    print(f'Median error: {median_trans} m, {median_rot} degrees.')
    print(f'Mean error: {mean_trans} m, {mean_rot} degrees.')
    print(f'Max error: {max_trans} m, {max_rot} degrees.')
    print(f'Min error: {min_trans} m, {min_rot} degrees.')

    run.log({"# Suc 5": successful_count_5, "# Suc 2": successful_count_2,
             "% Suc 5": success_rate_5, "% Suc 2": success_rate_2,
             "MedTrans": median_trans, "MedRot": median_rot, "AvgTrans": mean_trans, "AvgRot": mean_rot,
             "MaxTrans": max_trans, "MaxRot": max_rot, "MinTrans": min_trans, "MinRot": min_rot})

    vis_pose(pred_trans, gt_trans, errors_rot, vis_root)


if __name__ == "__main__":
    parser = config_parser()
    model = ModelParams(parser)
    optimization = OptimizationParams(parser)
    args = get_combined_args(parser)
    fix_seed(args.seed)
    run = wandb.init(
        project="RAP",
        config=args,
        name=args.run_name
    )
    refine(args, run)
