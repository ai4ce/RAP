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

import glob
import json
import os
from pathlib import Path
from typing import NamedTuple

import pandas as pd
from PIL import Image, ImageDraw
from plyfile import PlyData, PlyElement

from dataset_loaders.colmap_loader import *
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal, BasicPointCloud
from utils.sh_utils import SH2RGB


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    depth_path: str
    depth_params: dict
    mask_path: str
    image_path: str
    image_name: str
    width: int
    height: int


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str


def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1
    print(f"Center: {center}, Radius: {radius}")
    translate = -center

    return {"translate": translate, "radius": radius}


def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder,
                      depths_folder=None, depths_params=None, masks_folder=None):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        extr = cam_extrinsics[key]

        depth_path = None
        depth_params = None
        if depths_folder:
            depth_name = extr.name.replace(".color", "").replace(".jpg", ".png")
            depth_path = os.path.join(depths_folder, depth_name)
            if not os.path.exists(depth_path):
                depth_path = None
            if depths_params:
                depth_params = depths_params[os.path.splitext(depth_name)[0]]

        mask_path = None
        if masks_folder:
            mask_path = f"{masks_folder}/{extr.name}.png"
            if not os.path.exists(mask_path):
                mask_path = None

        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = qvec2rotmat(extr.qvec)
        T = np.array(extr.tvec)

        if intr.model == "SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, extr.name)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX,
                              depth_path=depth_path, depth_params=depth_params, mask_path=mask_path,
                              image_path=image_path, image_name=extr.name, width=width, height=height)
        cam_infos.append(cam_info)

    return cam_infos


def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def readColmapSceneInfo(path, images, depths=None, masks=None, eval=True, use_depth_loss=False, depth_is_inverted=False, use_masks=False, train_fraction=1):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0/images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0/cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except FileNotFoundError:
        cameras_extrinsic_file = os.path.join(path, "sparse/0/images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0/cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    image_dir = f"{path}/{images or 'images'}"
    depth_dir = None
    depths_params = None
    if use_depth_loss:
        depth_dir = f"{path}/{depths or 'depths'}"
        if not os.path.exists(depth_dir):
            raise ValueError("Depth loss is enabled but no depth maps found in the dataset.")
        if depth_is_inverted:
            with open(f"{path}/sparse/0/depth_params.json", "r") as f:
                depths_params = json.load(f)
            all_scales = np.array([depths_params[key]["scale"] for key in depths_params])
            if (all_scales > 0).sum():
                med_scale = np.median(all_scales[all_scales > 0])
            else:
                med_scale = 0
            for key in depths_params:
                depths_params[key]["med_scale"] = med_scale
    mask_dir = None
    if use_masks:
        mask_dir = f"{path}/{masks or 'masks'}"
        if not os.path.exists(mask_dir):
            raise ValueError("Masks are enabled but no mask images found in the dataset.")

    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics,
                                           images_folder=image_dir, depths_folder=depth_dir,
                                           depths_params=depths_params, masks_folder=mask_dir)
    cam_infos = sorted(cam_infos_unsorted, key=lambda x: x.image_name)

    train_cam_infos = []
    test_cam_infos = []
    if eval:
        tsvs = glob.glob(os.path.join(path, '*.tsv'))
        if len(tsvs) != 0:
            tsv = glob.glob(os.path.join(path, '*.tsv'))[0]
            print(f"Using {tsv} for train/test split.")
            files = pd.read_csv(tsv, sep='\t', usecols=['filename', 'split'], index_col='filename').dropna()
            train_split = files[files['split'] == 'train']
            test_split = files[files['split'] == 'test']
            train_cam_infos = [cam for cam in cam_infos if cam.image_name in train_split.index]
            test_cam_infos = [cam for cam in cam_infos if cam.image_name in test_split.index]
        elif os.path.exists(f"{path}/list_test.txt"):
            print(f"Using {path}/list_test.txt for train/test split.")
            with open(f"{path}/list_test.txt") as f:
                test_images = set(line.strip() for line in f)
            for cam in cam_infos:
                if cam.image_name in test_images:
                    test_cam_infos.append(cam)
                else:
                    train_cam_infos.append(cam)
        else:
            n_trains = int(len(cam_infos) * train_fraction)
            train_idxs = set(np.linspace(0, len(cam_infos) - 1, n_trains, dtype=int))
            for idx, cam in enumerate(cam_infos):
                if idx in train_idxs:
                    train_cam_infos.append(cam)
                else:
                    test_cam_infos.append(cam)
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    print(f"Train cameras: {len(train_cam_infos)} of {len(cam_infos)} ({len(train_cam_infos) / len(cam_infos):.2%})")
    print(f"Test cameras: {len(test_cam_infos)} of {len(cam_infos)} ({len(test_cam_infos) / len(cam_infos):.2%})")

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png", data_perturb=None,
                              split="train"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = w2c[:3, :3]
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)
            if idx != 0 and split == "train":
                image = add_perturbation(image, data_perturb, idx)

            im_data = np.array(image.convert("RGBA"))
            bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                                        image_path=image_path, image_name=image_name, width=image.size[0],
                                        height=image.size[1]))

    return cam_infos


def readNerfSyntheticInfo(path, white_background, eval, extension=".png", data_perturb=None):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension,
                                                data_perturb=data_perturb,
                                                split="train")  # [CameraInfo(id，fov，R，T，图片路径，图片，高，宽)...100长度]
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension,
                                               data_perturb=None, split="test")

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        # pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender": readNerfSyntheticInfo
}


def add_perturbation(img, perturbation, seed):
    if 'occ' in perturbation:
        draw = ImageDraw.Draw(img)
        np.random.seed(seed)
        left = np.random.randint(200, 400)
        top = np.random.randint(200, 400)
        for i in range(10):
            np.random.seed(10 * seed + i)
            random_color = tuple(np.random.choice(range(256), 3))
            draw.rectangle(((left + 20 * i, top), (left + 20 * (i + 1), top + 200)),
                           fill=random_color)

    if 'color' in perturbation:
        np.random.seed(seed)
        img_np = np.array(img) / 255.0  # H, W, 4
        s = np.random.uniform(0.8, 1.2, size=3)
        b = np.random.uniform(-0.2, 0.2, size=3)
        img_np[..., :3] = np.clip(s * img_np[..., :3] + b, 0, 1)
        img = Image.fromarray((255 * img_np).astype(np.uint8))

    return img
