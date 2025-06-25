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

import torch
from torch import nn

from utils.graphics_utils import getWorld2View2, get_projection_matrix, focal2fov, getWorld2View3
from utils.pose_utils import SE3_exp, interpolate_linear, interpolate_spline, interpolate_bezier


class Camera(nn.Module):
    def __init__(self, uid, colmap_id, image_name, R, T, K, FoVx, FoVy, image, scale=1, render_device="cuda", data_device="cuda", refine_pose=False):
        super().__init__()
        self.uid = uid
        self.colmap_id = colmap_id
        self.image_name = image_name
        self.R = R
        self.T = T
        self.K = K
        self.FoVx = FoVx
        self.FoVy = FoVy
        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device")
            self.data_device = torch.device("cuda")

        self.original_image = image.to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        self.zfar = 100.0
        self.znear = 0.01
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T), device=render_device)
        self.projection_matrix = get_projection_matrix(znear=self.znear, zfar=self.zfar,
                                                       fovX=self.FoVx, fovY=self.FoVy).to(render_device)
        self.full_proj_transform = self.projection_matrix @ self.world_view_transform
        self.camera_center = self.world_view_transform.inverse()[:3, 3]

        if refine_pose:
            self.world_view_transform = nn.Parameter(self.world_view_transform)
            self.optimizer = torch.optim.Adam([self.world_view_transform], lr=0.001)

    def update(self, *args):
        self.optimizer.step()
        self.optimizer.zero_grad()
        with torch.no_grad():
            u, s, v = torch.svd(self.world_view_transform[:3, :3])
            self.world_view_transform[:3, :3] = u @ v.T
            self.full_proj_transform = self.projection_matrix @ self.world_view_transform
            self.camera_center = torch.inverse(self.world_view_transform)[:3, 3]


class DeblurCamera(Camera):
    def __init__(self, uid, colmap_id, image_name, R, T, K, FoVx, FoVy,
                 image, mask, depth, depth_params, scale=1, data_device="cuda",
                 control_pts_num=2, deblur_mode="Linear"):
        super().__init__(uid, colmap_id, image_name, R, T, K, FoVx, FoVy, image, scale, data_device, refine_pose=False)
        if deblur_mode == "Linear":
            control_pts_num = 2
            self.interpolation_func = interpolate_linear
        elif deblur_mode == "Spline":
            control_pts_num = 2
            self.interpolation_func = interpolate_spline
        elif deblur_mode == "Bezier":
            self.interpolation_func = interpolate_bezier
        else:
            raise NotImplementedError(f"Deblur mode {deblur_mode} not implemented")

        self.mask = mask
        if mask is not None:
            self.mask = mask.to(self.data_device)

        self.depth = None
        self.depth_mask = None
        if depth_params is not None \
            and not (depth_params["scale"] < 0.2 * depth_params["med_scale"] or depth_params["scale"] > 5 * depth_params["med_scale"]):
            self.depth = depth.to(self.data_device) / float(2 ** 16)
            self.depth_mask = torch.ones_like(self.depth) if mask is None else self.mask.clone()
            self.depth[self.depth < 0] = 0
            if depth_params["scale"] > 0:
                self.depth = self.depth * depth_params["scale"] + depth_params["offset"]
            self.depth = self.depth
        elif depth is not None:
            self.depth = depth.to(self.data_device) / 1000
            self.depth_mask = ((depth != depth.max()) & (depth != depth.min())).float()
            if mask is not None:
                self.depth_mask *= self.mask

        self.world_view_transform = torch.tensor(getWorld2View2(R, T), device="cuda")
        self.projection_matrix = get_projection_matrix(znear=self.znear, zfar=self.zfar,
                                                       fovX=self.FoVx, fovY=self.FoVy).cuda()
        self.full_proj_transform = self.projection_matrix @ self.world_view_transform
        self.camera_center = self.world_view_transform.inverse()[:3, 3]

        # self.depth_factor = torch.nn.Parameter(
        #     torch.tensor([1, 0], dtype=torch.float, device="cuda", requires_grad=True))

        self.gaussian_trans = torch.nn.Parameter(
            (torch.zeros([control_pts_num, 6], device="cuda", requires_grad=True)))

        self.pose_optimizer = torch.optim.Adam([
            {'params': [self.gaussian_trans],
             'lr': 1.e-3, "name": "translation offset"},
        ], lr=0.0, eps=1e-15)
        # self.depth_optimizer = torch.optim.Adam([
        #     {'params': [self.depth_factor],
        #      'lr': 1e-3, "name": "depth factor"},
        # ], lr=0.0, eps=1e-15)

    def update(self, global_step):
        self.pose_optimizer.step()
        # self.depth_optimizer.step()
        self.pose_optimizer.zero_grad(set_to_none=True)
        # self.depth_optimizer.zero_grad(set_to_none=True)
        decay_rate_pose = 0.01
        pose_lrate = 1e-3
        lrate_decay = 200
        decay_steps = lrate_decay * 1000
        new_lrate_pose = pose_lrate * (decay_rate_pose ** (global_step / decay_steps))
        for param_group in self.pose_optimizer.param_groups:
            param_group['lr'] = new_lrate_pose
        # for param_group in self.depth_optimizer.param_groups:
        #     param_group['lr'] = new_lrate_pose

    def get_gaussian_trans(self, alpha=0):
        return self.interpolation_func(self.gaussian_trans, alpha)


class LearnableCamera:
    def __init__(
            self,
            uid,
            color,
            depth,
            gt_H_col,
            H_col,
            fx,
            fy,
            device="cuda:0",
    ):
        self.uid = uid
        self.device = device

        self.R = torch.tensor(H_col[:3, :3], device=device)
        self.T = torch.tensor(H_col[:3, 3], device=device)
        self.R_gt = torch.tensor(gt_H_col[:3, :3], device=device)
        self.T_gt = torch.tensor(gt_H_col[:3, 3], device=device)
        self.zfar = 100.0
        self.znear = 0.01
        self.original_image = color
        self.depth = depth
        _, h, w = color.shape
        self.intrinsic_matrix = torch.tensor([[fx, 0, w / 2], [0, fy, h / 2], [0, 0, 1]], device=device)
        self.fx = fx
        self.fy = fy
        self.cx = w / 2
        self.cy = h / 2
        self.FoVx = focal2fov(fx, w)
        self.FoVy = focal2fov(fy, h)
        self.image_height = h
        self.image_width = w

        self.cam_rot_delta = nn.Parameter(
            torch.zeros(3, requires_grad=True, device=device)
        )
        self.cam_trans_delta = nn.Parameter(
            torch.zeros(3, requires_grad=True, device=device)
        )

        self.projection_matrix = get_projection_matrix(znear=self.znear, zfar=self.zfar,
                                                       fovX=self.FoVx, fovY=self.FoVy).cuda()

        self.optimizer = torch.optim.Adam([self.cam_rot_delta, self.cam_trans_delta], lr=0.001)

    @property
    def world_view_transform(self):
        return getWorld2View3(self.R, self.T)

    @property
    def full_proj_transform(self):
        return self.projection_matrix @ self.world_view_transform

    @property
    def camera_center(self):
        return self.world_view_transform.inverse()[:3, 3]

    def update(camera, converged_threshold=1e-4):
        camera.optimizer.step()
        camera.optimizer.zero_grad()

        with torch.no_grad():
            tau = torch.cat([camera.cam_trans_delta, camera.cam_rot_delta], axis=0)

            T_w2c = torch.eye(4, device=tau.device)
            T_w2c[0:3, 0:3] = camera.R
            T_w2c[0:3, 3] = camera.T

            new_w2c = SE3_exp(tau) @ T_w2c

            new_R = new_w2c[0:3, 0:3]
            new_T = new_w2c[0:3, 3]

            converged = tau.norm() < converged_threshold
            camera.update_RT(new_R, new_T)

            camera.cam_rot_delta.data.fill_(0)
            camera.cam_trans_delta.data.fill_(0)
        return converged

    def update_RT(self, R, t):
        self.R = R.to(device=self.device)
        self.T = t.to(device=self.device)

    def clean(self):
        self.original_image = None
        self.depth = None

        self.cam_rot_delta = None
        self.cam_trans_delta = None


class CamParams:
    def __init__(self, camera, resolution, device):
        h, w, fx, fy = camera["height"], camera["width"], camera["fx"], camera["fy"]
        if 0 < resolution < 10:
            global_down = resolution
        else:  # should be a type that converts to float
            global_down = w / resolution
        scale = float(global_down)
        self.h, self.w, self.fx, self.fy = round(h / scale), round(w / scale), fx / scale, fy / scale
        self.K = torch.tensor([[self.fx, 0, self.w / 2],
                               [0, self.fy, self.h / 2],
                               [0, 0, 1]], device=device)
        self.FovX = focal2fov(self.fx, self.w)
        self.FovY = focal2fov(self.fy, self.h)


class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform.T
        self.full_proj_transform = full_proj_transform.T
        self.camera_center = self.world_view_transform.inverse()[:3, 3]
