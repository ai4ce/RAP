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
import copy
import time
from argparse import ArgumentParser
from os import makedirs

import cv2 as cv
import imageio
import torchvision
from torch.nn import functional as F
from tqdm import tqdm

from arguments import ModelParams, OptimizationParams, get_combined_args
from models.gs.gaussian_model import GaussianModel
from utils.general_utils import *
from utils.nvs_utils import generate_multi_views
from utils.scene import Scene

torch.set_float32_matmul_precision("high")


def render_interpolation(args, name, iteration, views, gaussians, background, select_idxs=None):
    if args.scene_name == "brandenburg":
        select_idxs = [88]  #
    elif args.scene_name == "sacre":
        select_idxs = [29]
    elif args.scene_name == "trevi":
        select_idxs = [55]
    else:
        select_idxs = random.sample(range(len(views)), 1)

    render_path = os.path.join(args.model_path, name, f"ours_{iteration}", f"intrinsic_dynamic_interpolate")
    render_path_gt = os.path.join(args.model_path, name, f"ours_{iteration}", f"intrinsic_dynamic_interpolate", "refer")
    makedirs(render_path, exist_ok=True)
    makedirs(render_path_gt, exist_ok=True)
    inter_weights = [i * 0.1 for i in range(0, 21)]
    select_views = [views[i] for i in select_idxs]
    for idx, view in enumerate(tqdm(select_views, desc="Rendering Interpolation")):
        gt = view.original_image[0:3, :, :].mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
        gt = cv.cvtColor(gt, cv.COLOR_RGB2BGR)
        cv.imwrite(os.path.join(render_path_gt, f"{select_idxs[idx]}_{view.colmap_id}.jpg"), gt,
                   [int(cv.IMWRITE_JPEG_QUALITY), 100])
        # torchvision.utils.save_image(view.original_image, os.path.join(render_path_gt, f"{select_idxs[idx]}_{view.colmap_id}" + ".png"))
        sub_s2d_inter_path = os.path.join(render_path, f"{select_idxs[idx]}_{view.colmap_id}")
        makedirs(sub_s2d_inter_path, exist_ok=True)
        for inter_weight in inter_weights:
            gaussians.colornet_inter_weight = inter_weight
            rendering = gaussians.render(view, args, background)["render"]
            rendering = rendering.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
            rendering = cv.cvtColor(rendering, cv.COLOR_RGB2BGR)
            cv.imwrite(os.path.join(sub_s2d_inter_path, f"{select_idxs[idx]}_{inter_weight:.2f}_{view.colmap_id}.jpg"),
                       rendering, [int(cv.IMWRITE_JPEG_QUALITY), 100])
    gaussians.colornet_inter_weight = 1.0


def render_multiview_video(args, name, train_views, gaussians, background):
    if args.scene_name == "brandenburg":
        format_idx = 11  # 4
        select_view_id = [12, 59, 305]
        length_view = 90 * 2
        appear_idxs = [313, 78]
        name = "train"
        view_appears = [train_views[i] for i in appear_idxs]

        # intrinsic_idxs=[0,1,2,3,4,5,7,8,9]
        # name="test"
        # view_intrinsics=[test_views[i] for i in intrinsic_idxs]
        views = [train_views[i] for i in select_view_id]
    elif args.scene_name == "sacre":
        format_idx = 38
        select_view_id = [753, 657, 595, 181, 699, ]  # 700
        length_view = 45 * 2

        appear_idxs = [350, 76]
        name = "train"
        view_appears = [train_views[i] for i in appear_idxs]

        # intrinsic_idxs=[6,12,15,17]
        # name="test"
        # view_intrinsics=[test_views[i] for i in intrinsic_idxs]
        views = [train_views[i] for i in select_view_id]
    elif args.scene_name == "trevi":
        format_idx = 17
        select_view_id = [408, 303, 79, 893, 395, 281]  # 700
        length_view = 45 * 2

        appear_idxs = [317, 495]

        name = "train"
        view_appears = [train_views[i] for i in appear_idxs]

        # intrinsic_idxs=[0,2,3,8,9,11]
        # name="test"
        # view_intrinsics=[test_views[i] for i in intrinsic_idxs]
        views = [train_views[i] for i in select_view_id]
    else:
        format_idx = random.randint(0, len(train_views) - 1)
        select_view_id = random.sample(range(len(train_views)), 4)
        views = [train_views[i] for i in select_view_id]
        length_view = 90 * 2
        appear_idxs = random.sample(range(len(train_views)), 2)
        view_appears = [train_views[i] for i in appear_idxs]

    for vid, view_appear in enumerate(tqdm(view_appears, desc="Rendering Video")):
        view_appear.image_height, view_appear.image_width = train_views[format_idx].image_height, train_views[
            format_idx].image_width
        view_appear.FoVx, view_appear.FoVy = train_views[format_idx].FoVx, train_views[format_idx].FoVy
        appear_idx = appear_idxs[vid]
        generated_views = generate_multi_views(views, view_appear, length=length_view)
        render_path = os.path.join(args.model_path, "demos", f"multiview_video",
                                   f"{name}_{appear_idx}_{view_appear.colmap_id}")
        makedirs(render_path, exist_ok=True)

        render_video_out = imageio.get_writer(f'{render_path}/000_mv_{name}_{appear_idx}_{view_appear.colmap_id}.mp4',
                                              mode='I', fps=60, codec='libx264', quality=10.0)
        _ = gaussians.render(view_appear, args, background, store_cache=True)["render"]
        for idx, view in enumerate(tqdm(generated_views, desc=f"{vid}")):
            view.camera_center = view_appear.camera_center
            rendering = gaussians.render(view, args, background, use_cache=True)["render"]
            rendering = rendering.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
            render_video_out.append_data(rendering)
            # cv.imwrite(os.path.join(render_path, f"{name}_{appear_idx}_{idx:05d}.jpg"),
            #            cv.cvtColor(rendering, cv.COLOR_RGB2BGR), [int(cv.IMWRITE_JPEG_QUALITY), 100])

        render_video_out.close()


def render_lego(model_path, name, iteration, views, view0, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, f"ours_{iteration}", f"renders")
    gts_path = os.path.join(model_path, name, f"ours_{iteration}", "gt")
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    _ = gaussians.render(view0, pipe=pipeline, bg_color=background, store_cache=True)["render"]
    for idx, view in enumerate(tqdm(views, desc="Rendering Reconstruction")):
        rendering = gaussians.render(view, pipeline, background, use_cache=True)["render"]
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, f'{idx:05d}.png'))
        torchvision.utils.save_image(gt, os.path.join(gts_path, f'{idx:05d}.png'))


def test_rendering_speed(views, gaussians, pipeline, background, use_cache=False):
    views = copy.deepcopy(views)
    length = len(views) - 1
    for idx in range(length):
        view = views[idx]
        view.original_image = F.interpolate(view.original_image.unsqueeze(0), size=(800, 800)).squeeze()
        view.image_height, view.image_width = 800, 800
    if not use_cache:
        _ = gaussians.render(views[0], pipeline, background)["render"]
        start_time = time.time()
        for idx in tqdm(range(length), desc="Testing Render Speed"):
            view = views[idx]
            _ = gaussians.render(view, pipeline, background)["render"]
        end_time = time.time()

        avg_rendering_speed = (end_time - start_time) / length
        print(f"Rendering speed: {1 / avg_rendering_speed:.2f} FPS")
        return avg_rendering_speed
    else:
        for i in range(length):
            views[i + 1].image_height, views[i + 1].image_width = view.image_height, view.image_width
        _ = gaussians.render(views[0], pipeline, background, store_cache=True)["render"]
        start_time = time.time()
        # for idx, view in enumerate(tqdm(views[1:], desc="Rendering progress")):
        for idx in tqdm(range(length), desc="Testing Render Speed"):
            view = views[idx + 1]
            _ = gaussians.render(view, pipeline, background, use_cache=True)["render"]
        end_time = time.time()
        avg_rendering_speed = (end_time - start_time) / length
        print(f"Rendering speed using cache: {1 / avg_rendering_speed:.2f} FPS")
        return avg_rendering_speed


def render_intrinsics(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, f"ours_{iteration}", "renders_intrinsic")
    makedirs(render_path, exist_ok=True)
    gaussians.colornet_inter_weight = 0.0
    for idx, view in enumerate(tqdm(views, desc="Rendering Intrinsics")):
        rendering = gaussians.render(view, pipeline, background)["render"]
        rendering = rendering.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
        rendering = cv.cvtColor(rendering, cv.COLOR_RGB2BGR)
        cv.imwrite(os.path.join(render_path, f'{idx:05d}.jpg'), rendering, [int(cv.IMWRITE_JPEG_QUALITY), 100])
        # torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
    gaussians.colornet_inter_weight = 1.0


def render_set(model_path, name, iteration, views, gaussians, pipeline, background,
               render_multi_view=False, render_s2d_inter=False):
    render_path = os.path.join(model_path, name, f"ours_{iteration}", "renders")
    gts_path = os.path.join(model_path, name, f"ours_{iteration}", "gt")
    if gaussians.use_features_mask:
        mask_path = os.path.join(model_path, name, f"ours_{iteration}", "masks")
        makedirs(mask_path, exist_ok=True)

    if render_multi_view:
        multi_view_path = os.path.join(model_path, name, f"ours_{iteration}", "multi_view")
    if render_s2d_inter:
        s2d_inter_path = os.path.join(model_path, name, f"ours_{iteration}", "intrinsic_dynamic_interpolate")
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    origin_views = copy.deepcopy(views)
    for idx, view in enumerate(tqdm(views, desc=f"Rendering {name}")):
        rendering = gaussians.render(view, pipeline, background)["render"]
        rendering = rendering.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
        rendering = cv.cvtColor(rendering, cv.COLOR_RGB2BGR)
        cv.imwrite(os.path.join(render_path, f'{idx:05d}.jpg'), rendering, [int(cv.IMWRITE_JPEG_QUALITY), 100])

        gt = view.original_image[0:3, :, :].mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
        gt = cv.cvtColor(gt, cv.COLOR_RGB2BGR)
        cv.imwrite(os.path.join(gts_path, f'{idx:05d}.jpg'), gt, [int(cv.IMWRITE_JPEG_QUALITY), 100])

        if gaussians.use_features_mask:
            tmask = gaussians.features_mask.repeat(1, 3, 1, 1)
            torchvision.utils.save_image(tmask, os.path.join(mask_path, f'{idx:05d}.png'))

        # torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        # torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

    if render_multi_view:
        # origin_views=copy.deepcopy(views)
        for idx, view in enumerate(tqdm(views, desc="Rendering Multiview")):
            sub_multi_view_path = os.path.join(multi_view_path, f"{idx}")
            makedirs(sub_multi_view_path, exist_ok=True)
            for o_idx, o_view in enumerate(tqdm(origin_views, desc=f"{idx}")):
                rendering = gaussians.render(view, pipeline, background, other_viewpoint_camera=o_view)["render"]
                rendering = rendering.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
                rendering = cv.cvtColor(rendering, cv.COLOR_RGB2BGR)
                cv.imwrite(os.path.join(sub_multi_view_path, f"{idx}_{o_idx}.jpg"), rendering,
                           [int(cv.IMWRITE_JPEG_QUALITY), 100])

    if render_s2d_inter and gaussians.color_net_type == "naive":
        views = origin_views
        inter_weights = [i * 0.1 for i in range(0, 21)]
        for idx, view in enumerate(tqdm(views, desc="Rendering Interpolation")):
            sub_s2d_inter_path = os.path.join(s2d_inter_path, f"{idx}")
            makedirs(sub_s2d_inter_path, exist_ok=True)
            for inter_weight in inter_weights:
                gaussians.colornet_inter_weight = inter_weight
                rendering = gaussians.render(view, pipeline, background)["render"]
                rendering = rendering.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
                rendering = cv.cvtColor(rendering, cv.COLOR_RGB2BGR)
                cv.imwrite(os.path.join(sub_s2d_inter_path, f"{idx}_{inter_weight:.2f}.jpg"), rendering,
                           [int(cv.IMWRITE_JPEG_QUALITY), 100])
        gaussians.colornet_inter_weight = 1.0


@torch.no_grad()
def render_sets(args):
    gaussians = GaussianModel(args)
    scene = Scene(args, gaussians, load_iteration=args.iteration, shuffle=False)

    bg_color = [1, 1, 1] if args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float, device=args.device)
    gaussians.set_eval(True)

    if args.scene_name == "lego":
        render_lego(args.model_path, "test", scene.loaded_iter, scene.test_cameras,
                    scene.train_cameras[0], gaussians, args, background)
        return
    if not args.skip_test:
        test_rendering_speed(scene.train_cameras, gaussians, args, background)
        if gaussians.color_net_type in ["naive"]:
            test_rendering_speed(scene.train_cameras, gaussians, args, background, use_cache=True)

        render_set(args.model_path, "test", scene.loaded_iter, scene.test_cameras, gaussians, args, background)

    if not args.skip_train:
        train_cameras = scene.train_cameras
        render_set(args.model_path, "train", scene.loaded_iter, train_cameras, gaussians, args, background)

        if gaussians.color_net_type in ["naive"]:
            render_intrinsics(args.model_path, "train", scene.loaded_iter, scene.train_cameras, gaussians,
                              args, background)

    if args.render_multiview_video:
        render_multiview_video(args, "train", scene.train_cameras, gaussians, background)
    if args.render_interpolation:
        # appearance tuning
        render_interpolation(args, "train", scene.loaded_iter, scene.train_cameras, gaussians, background)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)

    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--render_interpolation", action="store_true", default=False)
    parser.add_argument("--render_multiview_video", action="store_true", default=False)

    args = get_combined_args(parser)
    print("Rendering", args.model_path)

    fix_seed()

    render_sets(args)
