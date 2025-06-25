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
import cv2 as cv
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import transforms
from tqdm import tqdm

from utils.cameras import Camera, DeblurCamera
from utils.graphics_utils import fov2focal

WARNED = False


def loadCam(args, id, cam_info):
    orig_w, orig_h = cam_info.width, cam_info.height

    if 0 < args.resolution < 10:
        global_down = args.resolution
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                          "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

    scale = float(global_down)
    w, h = (round(orig_w / scale), round(orig_h / scale))

    intrinsic = torch.tensor([
        [fov2focal(cam_info.FovX, w), 0, w / 2],
        [0, fov2focal(cam_info.FovY, h), h / 2],
        [0, 0, 1]
    ], device=args.render_device)

    transform = transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ToTensor(),
    ])
    resized_image_rgb = transform(Image.open(cam_info.image_path))
    depth = None
    if cam_info.depth_path is not None:
        depth = cv.imread(cam_info.depth_path, cv.IMREAD_UNCHANGED)
        if depth.ndim != 2:
            depth = depth[..., 0]
        if depth.shape != (h, w):
            depth = cv.resize(depth, (w, h), interpolation=cv.INTER_NEAREST_EXACT)
        depth = torch.tensor(depth, dtype=torch.float)

    mask = None
    if cam_info.mask_path is not None:
        mask = cv.imread(cam_info.mask_path, cv.IMREAD_UNCHANGED)
        if mask.ndim != 2:
            mask = mask[..., 0]
        mask = (mask > 0).astype(np.float32)
        if mask.shape != (h, w):
            mask = cv.resize(mask, (w, h), interpolation=cv.INTER_NEAREST_EXACT)
        mask = torch.tensor(mask, dtype=torch.float)

    return DeblurCamera(uid=id, colmap_id=cam_info.uid, image_name=cam_info.image_name,
                        R=cam_info.R, T=cam_info.T, K=intrinsic, FoVx=cam_info.FovX, FoVy=cam_info.FovY, scale=scale,
                        image=resized_image_rgb, mask=mask, depth=depth, depth_params=cam_info.depth_params,
                        data_device=args.data_device, control_pts_num=args.bezier_order, deblur_mode=args.deblur_mode)


def cameraList_from_camInfos(cam_infos, args):
    camera_list = []

    for id, c in enumerate(tqdm(cam_infos, desc="Loading Cameras")):
        camera_list.append(loadCam(args, id, c))

    return camera_list


def camera_to_JSON(id, camera: Camera):
    Rt = np.eye(4, dtype=np.float64)
    Rt[:3, :3] = camera.R
    Rt[:3, 3] = camera.T

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id': id,
        'img_name': camera.image_name,
        'width': camera.width,
        'height': camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy': fov2focal(camera.FovY, camera.height),
        'fx': fov2focal(camera.FovX, camera.width)
    }
    return camera_entry
