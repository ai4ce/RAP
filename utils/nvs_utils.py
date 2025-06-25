import copy
import math
import os
import time

import cv2 as cv
import numpy as np
import torch
from torch.nn import functional as F
from multiprocessing.context import SpawnProcess
from multiprocessing import get_context
from kornia.geometry import rotation_matrix_to_quaternion, quaternion_to_rotation_matrix
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from models.gs.gaussian_model import GaussianModel
from utils.cameras import CamParams, Camera
from utils.general_utils import safe_path, search_for_max_iteration
from utils.graphics_utils import get_projection_matrix


# x rotation
rot_phi = lambda phi: np.array([
    [1, 0, 0, 0],
    [0, np.cos(phi), -np.sin(phi), 0],
    [0, np.sin(phi), np.cos(phi), 0],
    [0, 0, 0, 1]], dtype=np.float32)

# y rotation
rot_theta = lambda th: np.array([
    [np.cos(th), 0, -np.sin(th), 0],
    [0, 1, 0, 0],
    [np.sin(th), 0, np.cos(th), 0],
    [0, 0, 0, 1]], dtype=np.float32)

# z rotation
rot_psi = lambda psi: np.array([
    [np.cos(psi), -np.sin(psi), 0, 0],
    [np.sin(psi), np.cos(psi), 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]], dtype=np.float32)


def get_bbox(poses, d_max):
    """Determine the bounding box of the scene."""
    b_min = [poses[:, 0, 3].min() - d_max, poses[:, 1, 3].min() - d_max, poses[:, 2, 3].min() - d_max]
    b_max = [poses[:, 0, 3].max() + d_max, poses[:, 1, 3].max() + d_max, poses[:, 2, 3].max() + d_max]
    return b_min, b_max


def perturb_rotation(pose, theta, phi, psi=0):
    c2w = np.eye(4)
    c2w[:3, :4] = pose
    c2w = rot_phi(phi / 180. * np.pi) @ c2w
    c2w = rot_theta(theta / 180. * np.pi) @ c2w
    c2w = rot_psi(psi / 180. * np.pi) @ c2w
    return c2w[:3, :4]


def generate_LENS_poses_xy(poses, factor=5):
    translations = np.array([pose[:3, 3].numpy() for pose in poses])
    x_min, y_min, z_min = translations.min(axis=0)
    x_max, y_max, z_max = translations.max(axis=0)

    # 网格划分 (x, y 分量)
    x_values = np.linspace(x_min, x_max, int((len(poses) * factor) ** 0.5))
    y_values = np.linspace(y_min, y_max, int((len(poses) * factor) ** 0.5))
    virtual_translations = []
    for x in x_values:
        for y in y_values:
            # if not (x > 0 and y > 0):  # factor=7.3
            virtual_translations.append([x, y])

    # 使用最近的真实姿态补充 z 分量和旋转矩阵
    virtual_poses = []
    indexes = []
    for x, y in virtual_translations:
        # 找到最近的真实姿态
        distances = np.linalg.norm(translations[:, :2] - np.array([x, y]), axis=1)
        nearest_idx = np.argmin(distances)
        indexes.append(nearest_idx)
        nearest_pose = poses[nearest_idx]
        z = nearest_pose[:3, 3][2].item()  # 取最近姿态的 z 分量
        R = nearest_pose[:3, :3]  # 取最近姿态的旋转矩阵

        # 创建虚拟姿态
        virtual_pose = torch.cat((R, torch.tensor([x, y, z]).reshape(3, 1)), dim=1)
        virtual_poses.append(virtual_pose)

    return torch.stack(virtual_poses), indexes


def generate_LENS_poses_xz(poses, factor=9.5):
    translations = np.array([pose[:3, 3].numpy() for pose in poses])
    x_min, z_min, y_min = translations.min(axis=0)
    x_max, z_max, y_max = translations.max(axis=0)

    # 网格划分 (x, y 分量)
    x_values = np.linspace(x_min, x_max, int((len(poses) * factor) ** 0.5))
    y_values = np.linspace(y_min, y_max, int((len(poses) * factor) ** 0.5))
    virtual_translations = []
    for x in x_values:
        for y in y_values:
            if not (-5 < y < 50 and -5 < x < 35):
                virtual_translations.append([x, y])

    # 使用最近的真实姿态补充 z 分量和旋转矩阵
    virtual_poses = []
    indexes = []
    for x, y in virtual_translations:
        # 找到最近的真实姿态
        distances = np.linalg.norm(translations[:, [0, 2]] - np.array([x, y]), axis=1)
        nearest_idx = np.argmin(distances)
        indexes.append(nearest_idx)
        nearest_pose = poses[nearest_idx]
        z = nearest_pose[:3, 3][1].item()  # 取最近姿态的 z 分量
        R = nearest_pose[:3, :3]  # 取最近姿态的旋转矩阵

        # 创建虚拟姿态
        virtual_pose = torch.cat((R, torch.tensor([x, z, y]).reshape(3, 1)), dim=1)
        virtual_poses.append(virtual_pose)

    return torch.stack(virtual_poses), indexes


def filter_poses(poses_train, poses_val, min_pose_distance, min_rotation_angle):
    ts_val = poses_val[:, :3, 3]  # [N_val, 3]
    rs_val = poses_val[:, :3, :3]
    rs_train = poses_train[:, :3, :3]
    qs_val = rotation_matrix_to_quaternion(rs_val)  # [N_val, 4]
    qs_val = F.normalize(qs_val, dim=1)  # 归一化
    qs_train = rotation_matrix_to_quaternion(rs_train)  # [N_train, 4]
    qs_train = F.normalize(qs_train, dim=1)  # 归一化

    # 开始对训练集 pose 进行筛选
    filtered_indices = []
    for i in range(len(poses_train)):
        pose = poses_train[i]
        trans = pose[:3, 3]  # 获取平移部分

        # 计算每个训练姿态与验证姿态的平移距离
        differences = ts_val - trans  # [N_val, 3]
        distances = torch.linalg.vector_norm(differences, ord=2, dim=1) # [N_val]

        # 找到最近的验证姿态
        min_idx = torch.argmin(distances)
        min_distance = distances[min_idx]

        # 使用最近验证姿态的旋转进行角度计算
        q_train = qs_train[i]  # 训练集姿态的四元数
        q_val_nearest = qs_val[min_idx]  # 最近验证姿态的四元数
        theta = (q_train @ q_val_nearest).abs_().clamp_(-1.0, 1.0).acos_().mul_(2 * 180 / math.pi)

        # 如果满足距离和角度条件，将索引加入筛选列表
        if min_distance >= min_pose_distance and theta >= min_rotation_angle:
            filtered_indices.append(i)

    # 保留符合条件的训练集姿态、目标和RGB数据
    poses_filtered = poses_train[filtered_indices]
    return poses_filtered, filtered_indices, ts_val, qs_val


def perturb_poses(poses, perturb_func, trans, rot, b_min, b_max):
    poses_perturbed = torch.empty_like(poses)
    for i in range(len(poses)):
        poses_perturbed[i] = torch.tensor(perturb_func(poses[i].numpy(), trans, rot))
        for j in range(3):
            if poses_perturbed[i, j, 3] < b_min[j]:
                poses_perturbed[i, j, 3] = b_min[j]
            if poses_perturbed[i, j, 3] > b_max[j]:
                poses_perturbed[i, j, 3] = b_max[j]
    return poses_perturbed


def perturb_pose_with_criteria(pose, ts_val, qs_val, rand_trans, rand_rot, b_min, b_max, min_pose_distance, min_rotation_angle):
    for attempts in range(10000):
        perturbed_pose = torch.tensor(perturb_pose(pose, rand_trans, rand_rot))  # if want to compute on GPU, pass device=pose.device
        for j in range(3):
            if perturbed_pose[j, 3] < b_min[j]:
                perturbed_pose[j, 3] = b_min[j]
            if perturbed_pose[j, 3] > b_max[j]:
                perturbed_pose[j, 3] = b_max[j]

        perturbed_trans = perturbed_pose[:3, 3]
        perturbed_rot = perturbed_pose[:3, :3]

        differences = ts_val - perturbed_trans  # [N_val, 3]
        distances = torch.linalg.vector_norm(differences, ord=2, dim=1)  # [N_val]

        min_idx = torch.argmin(distances)
        min_distance = distances[min_idx]

        # Use the rotation of the nearest validation pose
        q_perturbed = rotation_matrix_to_quaternion(perturbed_rot)  # [1, 4]
        q_perturbed = F.normalize(q_perturbed, dim=0)  # [4]

        # Calculate quaternion dot product
        q_val_nearest = qs_val[min_idx]  # [4]
        theta = (q_perturbed @ q_val_nearest).abs_().clamp_(-1.0, 1.0).acos_().mul_(2 * 180 / math.pi)

        if min_distance >= min_pose_distance and theta >= min_rotation_angle:
            # Accept the perturbed pose if it meets the distance and angle criteria
            return perturbed_pose
    return None


def perturb_pose(pose, x, angle):
    """
    Inputs:
        pose: (3, 4)
        x: translation perturb range
        angle: rotation angle perturb range in degrees
    Outputs:
        new_c2w: (3, 4) new pose
    """
    # perturb rotation
    theta, phi, psi = np.random.uniform(-angle, angle, 3)  # in degrees, not really uniform
    new_c2w = perturb_rotation(pose, theta, phi, psi)

    # perturb translation
    trans_rand = np.random.uniform(-x, x, 3)  # random number of 3 axis pose perturbation
    new_c2w[:, 3] += trans_rand  # perturb pos between -1 to 1
    return new_c2w


def perturb_pose_xz(pose, x, angle):
    """
    Inputs:
        pose: (3, 4)
        x: translation perturb range
        angle: rotation angle perturb range in degrees
    Outputs:
        new_c2w: (3, 4) new pose
    """
    # perturb rotation
    theta, phi, psi = np.random.uniform(-angle, angle, 3)  # in degrees, not really uniform
    new_c2w = perturb_rotation(pose, theta, 0, 0)

    # perturb translation
    trans_rand = np.random.uniform(-x, x, 3)  # random number of 3 axis pose perturbation
    trans_rand[1] = 0
    new_c2w[:, 3] += trans_rand  # perturb pos between -1 to 1
    return new_c2w


def perturb_pose_uniform(pose, x, angle_max):
    """
    Inputs:
        pose: (3, 4)
        x: translation perturb range
        angle: rotation angle perturb range in degrees
    Outputs:
        new_c2w: (3, 4) new pose
    """
    # Randomly sample a rotation angle within the specified range
    theta = np.deg2rad(np.random.uniform(-angle_max, angle_max))  # Convert to radians

    # Sample a random axis (uniform on a sphere)
    axis = np.random.randn(3)
    axis /= np.linalg.norm(axis)  # Normalize to unit vector

    # Generate rotation matrix
    rotation = Rotation.from_rotvec(theta * axis).as_matrix()

    # Apply rotation and translation perturbations
    new_c2w = pose.copy()
    new_c2w[:3, :3] = rotation @ pose[:3, :3]

    # perturb translation
    trans_rand = np.random.uniform(-x, x, 3)  # random number of 3 axis pose perturbation
    new_c2w[:, 3] += trans_rand  # perturb pos between -1 to 1
    return new_c2w


def perturb_pose_uniform_and_sphere(pose, x, angle_max):
    """
    Inputs:
        pose: (3, 4) - Initial pose
        x: Translation perturb range
        angle_max: Maximum rotation angle in degrees
    Outputs:
        new_c2w: (3, 4) - New perturbed pose
    """
    # Randomly sample a rotation angle within the specified range
    theta = np.deg2rad(np.random.uniform(-angle_max, angle_max))  # Convert to radians

    # Sample a random axis (uniform on a sphere)
    axis = np.random.randn(3)
    axis /= np.linalg.norm(axis)  # Normalize to unit vector

    # Generate rotation matrix
    rotation = Rotation.from_rotvec(theta * axis).as_matrix()

    # Apply rotation and translation perturbations
    new_c2w = pose.copy()
    new_c2w[:3, :3] = rotation @ pose[:3, :3]

    # Perturb translation within a sphere
    r = np.random.uniform(0, x)  # Random radius
    theta = np.random.uniform(0, 2 * np.pi)  # Random azimuthal angle
    phi = np.random.uniform(0, np.pi)  # Random polar angle

    # Convert spherical coordinates to Cartesian
    trans_rand = np.array([
        r * np.sin(phi) * np.cos(theta),
        r * np.sin(phi) * np.sin(theta),
        r * np.cos(phi)
    ])
    new_c2w[:, 3] += trans_rand
    return new_c2w


class GaussianRenderer(SpawnProcess):
    def __init__(self, configs, cam_params, hw, dl, start_epoch, *args, **kwargs):
        super().__init__(*args, **kwargs)
        configs.deblur = False
        configs.use_depth_loss = False
        if configs.xz_plane_only:
            self.perturb_func = perturb_pose_xz
        elif configs.rvs_uniform_and_sphere:
            self.perturb_func = perturb_pose_uniform_and_sphere
        else:
            self.perturb_func = perturb_pose
        self.configs = configs
        self.cam_params: CamParams = cam_params
        self.hw = hw
        self.dl = dl
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=configs.render_device)[:, None, None]
        self.std = torch.tensor([0.229, 0.224, 0.225], device=configs.render_device)[:, None, None]
        self.queue = get_context("spawn").Queue(1)
        self.gaussians = None
        self.background = None
        self.imgs_orig = None
        self.img_names = None
        self.b_min = None
        self.b_max = None
        self.epoch = start_epoch

    def load_gaussians(self):
        self.gaussians = GaussianModel(self.configs)
        i = search_for_max_iteration(os.path.join(self.configs.model_path, "ckpts_point_cloud"))
        self.gaussians.load_ckpt_ply(os.path.join(self.configs.model_path, "ckpts_point_cloud", f"iteration_{i}", "point_cloud.ply"))
        bg_color = [1, 1, 1] if self.configs.white_background else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float, device=self.configs.render_device)
        self.gaussians.set_eval(True)

    @torch.no_grad()
    def render_set(self, name):
        imgs_normed = []
        imgs_rendered = []
        imgs_orig = []
        self.img_names = []
        poses = []
        render_path = os.path.join(self.configs.model_path, name, "ours_0", "renders")
        for batch_idx, (img_normed, pose, img_orig, img_name) in enumerate(tqdm(self.dl, desc="Rendering")):
            img_normed = img_normed[0]
            imgs_normed.append(img_normed)
            img_orig = img_orig[0]
            imgs_orig.append(img_orig)
            img_orig = img_orig.to(self.configs.render_device)
            pose = pose.reshape(3, 4)
            poses.append(pose)
            img_name = img_name[0]
            self.img_names.append(img_name)
            colmap_pose = torch.eye(4, dtype=torch.float)
            colmap_pose[:3, :4] = pose
            pose_inv = colmap_pose.inverse().numpy()
            R = pose_inv[:3, :3]
            T = pose_inv[:3, 3]
            view = Camera(uid=None, colmap_id=None, image_name=None, R=R, T=T, K=self.cam_params.K,
                          FoVx=self.cam_params.FovX, FoVy=self.cam_params.FovY, image=img_orig,
                          render_device=self.configs.render_device, data_device=self.configs.render_device)
            rendering: torch.Tensor = self.gaussians.render(view, self.configs, self.background)["render"]
            # resize
            rendering = F.interpolate(rendering[None], size=self.hw, mode='bilinear', align_corners=False)[0]
            normalized = rendering.sub(self.mean).div_(self.std)
            imgs_rendered.append(normalized.cpu())
            # save rendering
            if self.configs.vis_rvs:
                rendering_np = rendering.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
                rendering_np = cv.cvtColor(rendering_np, cv.COLOR_RGB2BGR)  # (480, 854, 3)
                cv.imwrite(safe_path(f"{render_path}/{img_name}.jpg"), rendering_np, [int(cv.IMWRITE_JPEG_QUALITY), 100])
        imgs_normed = torch.stack(imgs_normed)  # torch.Size([115, 3, 240, 427])
        imgs_rendered = torch.stack(imgs_rendered)  # torch.Size([115, 3, 240, 427])
        self.imgs_orig = torch.stack(imgs_orig)  # torch.Size([115, 3, 240, 427])
        poses = torch.stack(poses)  # torch.Size([115, 3, 4])
        self.b_min, self.b_max = get_bbox(poses, self.configs.d_max)
        return poses, imgs_normed, imgs_rendered

    @torch.no_grad()
    def render_perturbed_imgs(self, name, poses, indexes=None, disable_tqdm=False):
        rendered_imgs = []
        render_path = os.path.join(self.configs.model_path, name, f"ours_{self.epoch}", "virtual_renders")
        try:
            for batch_idx, pose in enumerate(tqdm(poses, desc=f"Rendering RVS @ Epoch {self.epoch}", disable=disable_tqdm)):
                colmap_pose = torch.eye(4, dtype=torch.float)
                colmap_pose[:3, :4] = pose
                pose_inv = colmap_pose.inverse().numpy()
                R = pose_inv[:3, :3]
                T = pose_inv[:3, 3]
                if indexes is not None:
                    ref_img = self.imgs_orig[indexes[batch_idx]]
                    img_name = self.img_names[indexes[batch_idx]]
                else:
                    ref_img = self.imgs_orig[batch_idx]
                    img_name = self.img_names[batch_idx]
                view = Camera(uid=None, colmap_id=None, image_name=None, R=R, T=T, K=self.cam_params.K,
                              FoVx=self.cam_params.FovX, FoVy=self.cam_params.FovY, image=ref_img,
                              render_device=self.configs.render_device, data_device=self.configs.render_device)
                if self.configs.no_appearance_augmentation:
                    self.gaussians.colornet_inter_weight = 1.0
                else:
                    self.gaussians.colornet_inter_weight = np.random.uniform(0, 2)
                rendering: torch.Tensor = self.gaussians.render(view, self.configs, self.background)["render"]
                # resize
                rendering = F.interpolate(rendering[None], size=self.hw, mode='bilinear', align_corners=False)[0]
                normalized = rendering.sub(self.mean).div_(self.std)
                rendered_imgs.append(normalized.cpu())
                if self.configs.vis_rvs:
                    rendering_np = rendering.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
                    rendering_np = cv.cvtColor(rendering_np, cv.COLOR_RGB2BGR)  # (480, 854, 3)
                    cv.imwrite(safe_path(f"{render_path}/{img_name}.jpg"), rendering_np, [int(cv.IMWRITE_JPEG_QUALITY), 100])
        except IndexError as e:
            print("Failed to sample appearance. Try decreasing RVS ranges.")
            raise e
        self.gaussians.colornet_inter_weight = 1.0
        return torch.stack(rendered_imgs)  # torch.Size([115, 3, 240, 427])

    def run(self):
        self.load_gaussians()
        poses, imgs_normed, imgs_rendered = self.render_set("train")
        self.queue.put((poses, imgs_normed, imgs_rendered))
        while True:
            begin = time.time()
            poses_perturbed = perturb_poses(poses, self.perturb_func, self.configs.rvs_trans, self.configs.rvs_rotation, self.b_min, self.b_max)
            imgs_perturbed = self.render_perturbed_imgs("train", poses_perturbed, disable_tqdm=True)
            end = time.time()
            print(f"RVS @ Epoch {self.epoch} Took {end - begin:.2f} s.")
            try:
                self.queue.put((poses_perturbed, imgs_perturbed))
                self.epoch += self.configs.rvs_refresh_rate
            except ValueError:
                break

class GaussianRendererWithAttempts(GaussianRenderer):
    @torch.no_grad()
    def render_perturbed_imgs(self, name, poses, indexes=None, disable_tqdm=False):
        perturbed_poses = []
        rendered_imgs = []
        render_path = os.path.join(self.configs.model_path, name, f"ours_{self.epoch}", "virtual_renders")
        for batch_idx, orig_pose in enumerate(tqdm(poses, desc=f"Rendering RVS @ Epoch {self.epoch}", disable=disable_tqdm)):
            best_pose = orig_pose
            if indexes is not None:
                ref_img = self.imgs_orig[indexes[batch_idx]]
                img_name = self.img_names[indexes[batch_idx]]
            else:
                ref_img = self.imgs_orig[batch_idx]
                img_name = self.img_names[batch_idx]
            for attempts in range(self.configs.max_attempts):
                pose = torch.tensor(self.perturb_func(orig_pose.numpy(), self.configs.rvs_trans, self.configs.rvs_rotation))
                for j in range(3):
                    if pose[j, 3] < self.b_min[j]:
                        pose[j, 3] = self.b_min[j]
                    if pose[j, 3] > self.b_max[j]:
                        pose[j, 3] = self.b_max[j]
                colmap_pose = torch.eye(4, dtype=torch.float)
                colmap_pose[:3, :4] = pose
                pose_inv = colmap_pose.inverse().numpy()
                R = pose_inv[:3, :3]
                T = pose_inv[:3, 3]
                view = Camera(uid=None, colmap_id=None, image_name=None, R=R, T=T, K=self.cam_params.K,
                              FoVx=self.cam_params.FovX, FoVy=self.cam_params.FovY, image=ref_img,
                              render_device=self.configs.render_device, data_device=self.configs.render_device)
                if self.configs.no_appearance_augmentation:
                    self.gaussians.colornet_inter_weight = 1.0
                else:
                    self.gaussians.colornet_inter_weight = np.random.uniform(0, 2)
                try:
                    rendering = self.gaussians.render(view, self.configs, self.background)["render"]
                except IndexError:
                    continue
                else:
                    best_pose = pose
                    break
            else:
                print(f"Warning: Could not render a perturbed {img_name}. Try decreasing RVS ranges.")
                colmap_pose = torch.eye(4, dtype=torch.float)
                colmap_pose[:3, :4] = orig_pose
                pose_inv = colmap_pose.inverse().numpy()
                R = pose_inv[:3, :3]
                T = pose_inv[:3, 3]
                view = Camera(uid=None, colmap_id=None, image_name=None, R=R, T=T, K=self.cam_params.K,
                              FoVx=self.cam_params.FovX, FoVy=self.cam_params.FovY, image=ref_img,
                              render_device=self.configs.render_device, data_device=self.configs.render_device)
                rendering = self.gaussians.render(view, self.configs, self.background)["render"]
            perturbed_poses.append(best_pose)
            rendering = F.interpolate(rendering[None], size=self.hw, mode='bilinear', align_corners=False)[0]
            normalized = rendering.sub(self.mean).div_(self.std)
            rendered_imgs.append(normalized.cpu())
            if self.configs.vis_rvs:
                rendering_np = rendering.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
                rendering_np = cv.cvtColor(rendering_np, cv.COLOR_RGB2BGR)  # (480, 854, 3)
                cv.imwrite(safe_path(f"{render_path}/{img_name}.jpg"), rendering_np,
                           [int(cv.IMWRITE_JPEG_QUALITY), 100])
        self.gaussians.colornet_inter_weight = 1.0
        return torch.stack(perturbed_poses), torch.stack(rendered_imgs)  # torch.Size([115, 3, 240, 427])

    def run(self):
        self.load_gaussians()
        poses, imgs_normed, imgs_rendered = self.render_set("train")
        self.queue.put((poses, imgs_normed, imgs_rendered))
        while True:
            print(f"RVS @ Epoch {self.epoch}...")
            begin = time.time()
            poses_perturbed, imgs_perturbed = self.render_perturbed_imgs("train", poses, disable_tqdm=True)
            end = time.time()
            print(f"RVS @ Epoch {self.epoch} Took {end - begin:.2f} s.")
            try:
                self.queue.put((poses_perturbed, imgs_perturbed))
                self.epoch += self.configs.rvs_refresh_rate
            except ValueError:
                break


class GaussianRendererWithBrisqueAttempts(GaussianRendererWithAttempts):
    def __init__(self, configs, cam_params, hw, dl, start_epoch, *args, **kwargs):
        super().__init__(configs, cam_params, hw, dl, start_epoch, *args, **kwargs)
        self.brisque = None

    @torch.no_grad()
    def render_perturbed_imgs(self, name, poses, indexes=None, disable_tqdm=False):
        if self.brisque is None:
            from brisque import BRISQUE
            self.brisque = BRISQUE()
        perturbed_poses = []
        rendered_imgs = []
        best_view = None
        best_rendering = None
        render_path = os.path.join(self.configs.model_path, name, f"ours_{self.epoch}", "virtual_renders")
        for batch_idx, orig_pose in enumerate(tqdm(poses, desc=f"Rendering RVS @ Epoch {self.epoch}", disable=disable_tqdm)):
            min_score = float('inf')
            best_pose = orig_pose
            if indexes is not None:
                ref_img = self.imgs_orig[indexes[batch_idx]]
                img_name = self.img_names[indexes[batch_idx]]
            else:
                ref_img = self.imgs_orig[batch_idx]
                img_name = self.img_names[batch_idx]
            for attempts in range(self.configs.max_attempts):
                pose = torch.tensor(self.perturb_func(orig_pose.numpy(), self.configs.rvs_trans, self.configs.rvs_rotation))
                for j in range(3):
                    if pose[j, 3] < self.b_min[j]:
                        pose[j, 3] = self.b_min[j]
                    if pose[j, 3] > self.b_max[j]:
                        pose[j, 3] = self.b_max[j]
                colmap_pose = torch.eye(4, dtype=torch.float)
                colmap_pose[:3, :4] = pose
                pose_inv = colmap_pose.inverse().numpy()
                R = pose_inv[:3, :3]
                T = pose_inv[:3, 3]
                view = Camera(uid=None, colmap_id=None, image_name=None, R=R, T=T, K=self.cam_params.K,
                              FoVx=self.cam_params.FovX, FoVy=self.cam_params.FovY, image=ref_img,
                              render_device=self.configs.render_device, data_device=self.configs.render_device)
                self.gaussians.colornet_inter_weight = 1.0
                rendering = self.gaussians.render(view, self.configs, self.background,
                                                  store_cache=attempts == 0, use_cache=attempts > 0)["render"]
                rendering = F.interpolate(rendering[None], size=self.hw, mode='bilinear', align_corners=False)[0]
                score = self.brisque.score(rendering.permute(1, 2, 0).cpu())
                if score < min_score:
                    min_score = score
                    best_pose = pose
                    best_view = view
                    best_rendering = rendering
                    if score < self.configs.brisque_threshold:
                        break
                if self.configs.vis_rvs:
                    rendering_np = rendering.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
                    rendering_np = cv.cvtColor(rendering_np, cv.COLOR_RGB2BGR)  # (480, 854, 3)
                    cv.imwrite(safe_path(f"{render_path}/{img_name}_deny_{score:.2f}.jpg"), rendering_np,
                               [int(cv.IMWRITE_JPEG_QUALITY), 100])
            # else:
            #     print(f"Warning: Could not find a valid perturbed pose at {img_name}. Min score: {min_score:.2f}")
            perturbed_poses.append(best_pose)
            if not self.configs.no_appearance_augmentation:
                self.gaussians.colornet_inter_weight = np.random.uniform(0, 2)
                best_rendering = self.gaussians.render(best_view, self.configs, self.background)["render"]
                best_rendering = F.interpolate(best_rendering[None], size=self.hw, mode='bilinear', align_corners=False)[0]
            normalized = best_rendering.sub(self.mean).div_(self.std)
            rendered_imgs.append(normalized.cpu())
            if self.configs.vis_rvs:
                rendering_np = best_rendering.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
                rendering_np = cv.cvtColor(rendering_np, cv.COLOR_RGB2BGR)  # (480, 854, 3)
                cv.imwrite(safe_path(f"{render_path}/{img_name}_acc_{min_score:.2f}.jpg"), rendering_np,
                           [int(cv.IMWRITE_JPEG_QUALITY), 100])
        self.gaussians.colornet_inter_weight = 1.0
        return torch.stack(perturbed_poses), torch.stack(rendered_imgs)  # torch.Size([115, 3, 240, 427])


def slerp(q0, q1, t):
    dot = torch.dot(q0, q1)
    dot = torch.clamp(dot, -1.0, 1.0)

    if dot < 0.0:
        q1 = -q1
        dot = -dot

    if dot > 0.9995:
        result = q0 + t * (q1 - q0)
        return result / torch.linalg.vector_norm(result, ord=2)

    theta_0 = torch.acos(dot)
    theta = theta_0 * t
    sin_theta_0 = torch.sin(theta_0)
    sin_theta = torch.sin(theta)

    s0 = torch.sin(theta_0 - theta) / sin_theta_0
    s1 = sin_theta / sin_theta_0
    return (s0 * q0) + (s1 * q1)


def interpolate_world_view_transform(transform0, transform1, t):
    rot0, trans0 = transform0[:3, :3], transform0[:3, 3]
    rot1, trans1 = transform1[:3, :3], transform1[:3, 3]

    quat0 = rotation_matrix_to_quaternion(rot0)
    quat1 = rotation_matrix_to_quaternion(rot1)
    quat_interp = slerp(quat0, quat1, t)

    trans_interp = (1 - t) * trans0 + t * trans1

    new_rot = quaternion_to_rotation_matrix(quat_interp)
    new_transform = torch.eye(4, device=transform0.device)
    new_transform[:3, :3] = new_rot
    new_transform[:3, 3] = trans_interp
    return new_transform


def interpolate_views(src_view, dst_view, view_intrinsic, length):
    results = []

    src_view.FoVy = np.tan(
        (np.arctan(view_intrinsic.FoVy / 2) / np.arctan(view_intrinsic.FoVx / 2)) * np.arctan(src_view.FoVx / 2)) * 2
    dst_view.FoVy = np.tan(
        (np.arctan(view_intrinsic.FoVy / 2) / np.arctan(view_intrinsic.FoVx / 2)) * np.arctan(dst_view.FoVx / 2)) * 2
    for i in range(length):
        view_temp = copy.deepcopy(view_intrinsic)

        view_temp.FoVy = src_view.FoVy * (1 - i / length) + dst_view.FoVy * (i / length)
        view_temp.FoVx = src_view.FoVx * (1 - i / length) + dst_view.FoVx * (i / length)

        view_temp.projection_matrix = get_projection_matrix(0.01, 100.0, view_temp.FoVx, view_temp.FoVy).to(src_view.world_view_transform.device)

        view_temp.world_view_transform = interpolate_world_view_transform(src_view.world_view_transform,
                                                                          dst_view.world_view_transform, i / length)

        view_temp.full_proj_transform = view_temp.projection_matrix @ view_temp.world_view_transform
        view_temp.camera_center = view_temp.world_view_transform.inverse()[:3, 3]
        results.append(view_temp)
    return results


def generate_multi_views(views, view_intrinsic, length=60):
    generated_views = []
    for i in range(len(views) - 1):
        views_temp = interpolate_views(views[i], views[i + 1], view_intrinsic, length)
        generated_views.extend(views_temp)
    return generated_views
