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
import os

import numpy as np
from gsplat import rasterization
from plyfile import PlyData, PlyElement
from sklearn.neighbors import NearestNeighbors

from models.gs.color_features_net import *
from models.gs.feature_maps_generators import *
from models.gs.feature_maps_projection import *
from models.gs.feature_maps_sample import *
from utils.general_utils import get_expon_lr_func, build_rotation, safe_path
from utils.general_utils import strip_symmetric, build_scaling_rotation
from utils.graphics_utils import BasicPointCloud
from utils.sh_utils import RGB2SH


class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = torch.logit

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, args):
        self.device = torch.device(args.render_device)
        self.active_sh_degree = 0
        self.max_sh_degree = args.sh_degree
        self._xyz = torch.empty(0)

        self._features_intrinsic = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()
        self.eval_mode = False

        self.use_colors_precomp = args.use_colors_precomp
        self._pre_comp_color = torch.empty(0)
        self.coord_scale = args.coord_scale
        self.map_num = args.map_num
        self.use_xw_init_box_coord = args.use_xw_init_box_coord
        self.feature_maps_combine = args.feature_maps_combine
        self.use_xw_init_box_coord = args.use_xw_init_box_coord
        self.use_without_adaptive = args.use_without_adaptive
        self.use_features_mask = args.use_features_mask

        self.use_kmap_pjmap, self.use_okmap = False, False
        if args.use_kmap_pjmap:
            self.use_kmap_pjmap = args.use_kmap_pjmap
        elif args.use_okmap:
            self.use_okmap = args.use_okmap
        else:
            raise NotImplementedError

        if args.map_generator_type == "unet":
            self.map_generator_orig = UNet(**args.map_generator_params).to(self.device)
            if args.compile:
                self.map_generator = torch.compile(self.map_generator_orig)
            else:
                self.map_generator = self.map_generator_orig
        else:
            raise NotImplementedError

        self.use_color_net = args.use_color_net
        if self.use_color_net:
            self.color_net_type = args.color_net_type
            self.colornet_inter_weight = 1.0
            self.projection_feature_weight = 1.0
            if args.color_net_type == "naive":
                self.color_net_orig = ColorNet(**args.color_net_params).to(self.device)
                if args.compile:
                    self.color_net = torch.compile(self.color_net_orig)
                else:
                    self.color_net = self.color_net_orig
            else:
                raise NotImplementedError

    def __len__(self):
        return self._xyz.shape[0]

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_xyz_dealt(self):
        return self._xyz_dealt

    @property
    def get_features(self):
        return self._features_intrinsic

    @property
    def get_features_dealt(self):
        return self._features_dealt

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_opacity_dealt(self):
        return self.opacity_activation(self._opacity_dealt)

    @property
    def get_colors(self):
        return self._pre_comp_color

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float, downsample_resolution=1):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(pcd.points, dtype=torch.float, device=self.device)

        fused_color = RGB2SH(torch.tensor(pcd.colors, dtype=torch.float, device=self.device))

        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2),
                               dtype=torch.float, device=self.device)
        features[:, :3, 0] = fused_color
        # features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation:", fused_point_cloud.shape[0])

        model = NearestNeighbors(n_neighbors=4, metric="euclidean").fit(pcd.points)
        distances, _ = model.kneighbors(pcd.points)
        scales = torch.tensor(distances[:, 1:], dtype=torch.float, device=self.device).square_().mean(dim=-1).sqrt_().log_()[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), dtype=torch.float, device=self.device)
        rots[:, 0] = 1
        opacities = torch.full((fused_point_cloud.shape[0], 1), 0.1, dtype=torch.float, device=self.device).logit_()

        self._xyz = nn.Parameter(fused_point_cloud)
        self._features_intrinsic = nn.Parameter(features[:, :, :].transpose(1, 2).contiguous())
        self._scaling = nn.Parameter(scales)
        self._rotation = nn.Parameter(rots)
        self._opacity = nn.Parameter(opacities)
        self.max_radii2D = torch.zeros(len(self), device=self.device)

        if self.use_kmap_pjmap and self.use_xw_init_box_coord:
            norm_xyz = (self._xyz - self._xyz.min(dim=0)[0]) / (self._xyz.max(dim=0)[0] - self._xyz.min(dim=0)[0])
            norm_xyz = (norm_xyz - 0.5) * 2
            box_coord = torch.rand(size=(len(self), self.map_num, 2), device=self.device)

            for i in range(self.map_num - 1):
                rand_weight = torch.rand((2, 3), device=self.device)
                rand_weight = rand_weight / rand_weight.sum(dim=-1).unsqueeze(1)
                box_coord[:, i, :] = torch.einsum('bi,ni->nb', rand_weight, norm_xyz) * self.coord_scale
                logging.info(f"rand sample coordinate weight: {rand_weight}")

            box_coord[:, -1, :] = box_coord[:, -1, :] * 0 + 256 * 2 * 2 / downsample_resolution
            self.box_coord = nn.Parameter(box_coord)

        elif self.use_okmap and self.use_xw_init_box_coord:
            norm_xyz = (self._xyz - self._xyz.min(dim=0)[0]) / (self._xyz.max(dim=0)[0] - self._xyz.min(dim=0)[0])
            norm_xyz = (norm_xyz - 0.5) * 2
            box_coord = torch.rand((len(self), self.map_num, 2), device=self.device)
            for i in range(self.map_num - 1):
                rand_weight = torch.rand((2, 3), device=self.device)
                rand_weight = rand_weight / rand_weight.sum(dim=-1).unsqueeze(1)
                box_coord[:, i, :] = torch.einsum('bi,ni->nb', rand_weight, norm_xyz) * self.coord_scale
                logging.info(f"rand sample coordinate weight: {rand_weight}")

            self.box_coord = nn.Parameter(box_coord)

        else:
            raise NotImplementedError

        if self.use_kmap_pjmap and self.use_without_adaptive:
            self.box_coord1 = nn.Parameter(box_coord[:, -1, :])
            self.box_coord2 = nn.Parameter(box_coord[:, :-1, :])

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((len(self), 1), device=self.device)
        self.denom = torch.zeros((len(self), 1), device=self.device)

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},

            {'params': [self._features_intrinsic], 'lr': training_args.feature_lr / 20.0, "name": "f_intr"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]
        self.prune_params_names = {"xyz", "f_intr", "opacity", "scaling", "rotation"}

        if self.use_kmap_pjmap or self.use_okmap:
            l.extend([
                {'params': self.map_generator.parameters(), 'lr': training_args.map_generator_lr,
                 "name": "map_generator"},
            ])

            if self.use_kmap_pjmap and self.use_without_adaptive:
                l.extend([
                    {'params': self.box_coord1, 'lr': training_args.box_coord_lr, "name": "box_coord"},
                ])
            else:
                l.extend([
                    {'params': self.box_coord, 'lr': training_args.box_coord_lr, "name": "box_coord"},
                ])
            self.prune_params_names.add("box_coord")

        if self.use_color_net:
            l.extend([
                {'params': self.color_net.parameters(), 'lr': training_args.color_net_lr, "name": "color_net"},
            ])

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init * self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final * self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        if self.use_kmap_pjmap or self.use_okmap:
            self.map_generator_scheduler_args = get_expon_lr_func(lr_init=training_args.map_generator_lr * 1,
                                                                  lr_final=training_args.map_generator_lr * 0.1,
                                                                  lr_delay_mult=0.1,
                                                                  max_steps=training_args.position_lr_max_steps)

            self.box_coord_scheduler_args = get_expon_lr_func(lr_init=training_args.box_coord_lr * 1,
                                                              lr_final=training_args.box_coord_lr * 0.01,
                                                              lr_delay_mult=0.1,
                                                              max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration, warm_up_iter=0):
        """ Learning rate scheduling per step """
        lrs = []
        length_update = 1
        if (self.use_kmap_pjmap or self.use_okmap) and iteration > warm_up_iter:
            length_update += 2

        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                lrs.append(lr)

            elif (self.use_kmap_pjmap or self.use_okmap) and iteration > warm_up_iter:
                if param_group["name"] == "box_coord":
                    lr = self.box_coord_scheduler_args(iteration)
                    param_group["lr"] = lr
                    lrs.append(lr)
                elif param_group["name"] == "map_generator":
                    lr = self.map_generator_scheduler_args(iteration)
                    param_group["lr"] = lr
                    lrs.append(lr)

            if len(lrs) == length_update:
                return lrs

    def set_learning_rate(self, name, lr):
        """ Learning rate scheduling per step """
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == name:
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        # for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
        #     l.append('f_dc_{}'.format(i))
        for i in range(self._features_intrinsic.shape[1] * self._features_intrinsic.shape[2]):
            l.append(f'f_intr_{i}')
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append(f'scale_{i}')
        for i in range(self._rotation.shape[1]):
            l.append(f'rot_{i}')

        return l

    @torch.no_grad()
    def save_ckpt_ply(self, path):
        xyz = self._xyz.cpu().numpy()
        normals = np.zeros_like(xyz)

        f_intr = self._features_intrinsic.transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.cpu().numpy()
        scale = self._scaling.cpu().numpy()
        rotation = self._rotation.cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)

        attributes = np.concatenate((xyz, normals, f_intr, opacities, scale, rotation), axis=1)

        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(safe_path(path))
        root_path = os.path.dirname(path)
        other_attributes_dict = {"non": torch.rand(1, 1)}
        if self.use_kmap_pjmap or self.use_okmap:
            torch.save(self.map_generator_orig.state_dict(), safe_path(os.path.join(root_path, "map_generator.pth")))

            if self.use_kmap_pjmap and self.use_without_adaptive:
                other_attributes_dict["box_coord1"] = self.box_coord1
                other_attributes_dict["box_coord2"] = self.box_coord2
            else:
                other_attributes_dict["box_coord"] = self.box_coord

        if self.use_color_net:
            torch.save(self.color_net_orig.state_dict(), safe_path(os.path.join(root_path, "color_net.pth")))

        torch.save(other_attributes_dict, safe_path(os.path.join(root_path, "other_attributes_dict.pth")))

    def reset_opacity(self):
        opacities_new = torch.min(self.get_opacity, torch.full_like(self._opacity, 0.01)).logit_()
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ckpt_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])), axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_intr_")]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)  3, (self.max_sh_degree + 1) ** 2 - 1)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device=self.device))

        self._features_intrinsic = nn.Parameter(
            torch.tensor(features_extra, dtype=torch.float, device=self.device).transpose(1, 2).contiguous())
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device=self.device))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device=self.device))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device=self.device))
        root_path = os.path.dirname(path)

        self.active_sh_degree = self.max_sh_degree
        other_attributes_dict = torch.load(os.path.join(root_path, "other_attributes_dict.pth"),
                                           map_location=self.device)

        if self.use_kmap_pjmap or self.use_okmap:
            map_generator_state_dict = torch.load(os.path.join(root_path, "map_generator.pth"), map_location=self.device)
            try:
                self.map_generator.load_state_dict(map_generator_state_dict)
            except RuntimeError:
                self.map_generator_orig.load_state_dict(map_generator_state_dict)

            if self.use_kmap_pjmap and self.use_without_adaptive:
                self.box_coord1 = other_attributes_dict["box_coord1"]
                self.box_coord2 = other_attributes_dict["box_coord2"]
            else:
                self.box_coord = other_attributes_dict["box_coord"]

        if self.use_color_net:
            color_net_state_dict = torch.load(os.path.join(root_path, "color_net.pth"), map_location=self.device)
            try:
                self.color_net.load_state_dict(color_net_state_dict)
            except RuntimeError:
                self.color_net_orig.load_state_dict(color_net_state_dict)

    def forward(self, viewpoint_camera, store_cache=False):
        img = viewpoint_camera.original_image.to(self.device)
        camera_center = viewpoint_camera.camera_center
        _xyz, _opacity, _features_intrinsic = self._xyz, self._opacity, self._features_intrinsic

        appearance_features = torch.empty(0)

        view_direction = (_xyz - camera_center.expand(_xyz.shape[0], -1))
        view_direction = view_direction / (view_direction.norm(dim=1, keepdim=True) + 1e-5)
        if self.use_kmap_pjmap:
            out_gen = self.map_generator(img.unsqueeze(0), eval_mode=self.eval_mode)
            feature_maps = out_gen["feature_maps"]

            if self.use_features_mask:
                self.features_mask = out_gen["mask"]
            if self.use_kmap_pjmap:
                if self.use_without_adaptive:
                    box_coord1, box_coord2 = self.box_coord1, self.box_coord2
                else:
                    box_coord1, box_coord2 = self.box_coord[:, -1, :], self.box_coord[:, :self.map_num - 1, :]
                feature_maps = feature_maps.reshape(self.map_num, -1, feature_maps.shape[-2], feature_maps.shape[-1])
                if self.use_xw_init_box_coord:
                    appearance_features_sampled = torch.empty(0)

                    if self.map_num - 1 > 0:
                        appearance_features_sampled, self.map_pts_norm = sample_from_feature_maps(
                            feature_maps[:self.map_num - 1, ...], box_coord2, self.coord_scale,
                            self.feature_maps_combine)
                    appearance_features_projected, project_mask = project2d(_xyz, viewpoint_camera.world_view_transform,
                                                                            viewpoint_camera.K, box_coord1,
                                                                            feature_maps[-1, ...].unsqueeze(0))
                else:
                    raise NotImplementedError

                appearance_features = torch.cat((appearance_features_sampled, appearance_features_projected), dim=1)

        if self.use_color_net:
            if self.use_colors_precomp:
                if self.color_net_type == "naive":
                    self._pre_comp_color = self.color_net(_xyz, _features_intrinsic, appearance_features, view_direction,
                                                          inter_weight=self.colornet_inter_weight,
                                                          store_cache=store_cache)
                else:
                    raise NotImplementedError
            else:
                if self.color_net_type == "naive":
                    self._features_dealt = self.color_net(_xyz, _features_intrinsic, appearance_features, view_direction,
                                                          inter_weight=self.colornet_inter_weight,
                                                          store_cache=store_cache)
                else:
                    raise NotImplementedError
        else:
            self._features_dealt = _features_intrinsic

        self._opacity_dealt = _opacity
        self._point_features = appearance_features
        self.view_direction = view_direction
        self._xyz_dealt = _xyz

    def forward_cache(self, viewpoint_camera):
        camera_center = viewpoint_camera.camera_center
        _xyz = self._xyz
        view_direction = (_xyz - camera_center.expand(_xyz.shape[0], -1))
        view_direction = view_direction / (view_direction.norm(dim=1, keepdim=True) + 1e-5)

        if self.use_colors_precomp:
            if self.color_net_type == "naive":
                self._pre_comp_color = self.color_net.forward_cache(_xyz, view_direction)
            else:
                raise NotImplementedError

        self.view_direction = view_direction
        self._xyz_dealt = _xyz

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor)
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] not in self.prune_params_names:
                continue
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask]))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask])
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]

        self._features_intrinsic = optimizable_tensors["f_intr"]

        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        if self.use_kmap_pjmap or self.use_okmap:
            if self.use_kmap_pjmap and self.use_without_adaptive:
                self.box_coord1 = optimizable_tensors["box_coord"]
                self.box_coord2 = self.box_coord2[valid_points_mask]
            else:
                self.box_coord = optimizable_tensors["box_coord"]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] not in self.prune_params_names:
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"],
                                                     torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"],
                                                        torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_tensor):

        optimizable_tensors = self.cat_tensors_to_optimizer(new_tensor)
        self._xyz = optimizable_tensors["xyz"]

        self._features_intrinsic = optimizable_tensors["f_intr"]

        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        if self.use_kmap_pjmap or self.use_okmap:
            if self.use_kmap_pjmap and self.use_without_adaptive:
                self.box_coord1 = optimizable_tensors["box_coord"]
                self.box_coord2 = torch.cat((self.box_coord2, new_tensor["box_coord2"]), dim=0)
            else:
                self.box_coord = optimizable_tensors["box_coord"]

        self.xyz_gradient_accum = torch.zeros((len(self), 1), device=self.device)
        self.denom = torch.zeros((len(self), 1), device=self.device)
        self.max_radii2D = torch.zeros(len(self), device=self.device)

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        new_tensor = {}
        n_init_points = len(self)
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros(n_init_points, dtype=torch.float, device=self.device)
        padded_grad[:grads.shape[0]] = grads.squeeze()
        large_grad_mask = padded_grad >= grad_threshold
        large_scale_mask = torch.max(self.get_scaling, dim=1).values > self.percent_dense * scene_extent
        selected_pts_mask = large_grad_mask & large_scale_mask

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), dtype=torch.float, device=self.device)
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_tensor["xyz"] = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self._xyz[selected_pts_mask].repeat(N, 1)
        new_tensor["scaling"] = self.scaling_inverse_activation(
            self.get_scaling[selected_pts_mask] / (0.8 * N)).repeat(N, 1)
        new_tensor["opacity"] = self._opacity[selected_pts_mask].repeat(N, 1)
        new_tensor["rotation"] = self._rotation[selected_pts_mask].repeat(N, 1)

        new_tensor["f_intr"] = self._features_intrinsic[selected_pts_mask].repeat(N, 1, 1)

        if self.use_kmap_pjmap or self.use_okmap:
            if self.use_kmap_pjmap and self.use_without_adaptive:
                new_tensor["box_coord"] = self.box_coord1[selected_pts_mask].repeat(N, 1)
                new_tensor["box_coord2"] = self.box_coord2[selected_pts_mask].repeat(N, 1, 1)
            else:
                new_tensor["box_coord"] = self.box_coord[selected_pts_mask].repeat(N, 1, 1)

        self.densification_postfix(new_tensor)
        print(f"{selected_pts_mask.sum()} points split and pruned")

        prune_filter = torch.cat(
            (selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), dtype=torch.bool, device=self.device)))

        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        large_grad_mask = torch.norm(grads, dim=-1) >= grad_threshold
        tiny_scale_mask = torch.max(self.get_scaling, dim=1).values <= self.percent_dense * scene_extent
        selected_pts_mask = large_grad_mask & tiny_scale_mask

        new_tensor = {"xyz": self._xyz[selected_pts_mask],
                      "f_intr": self._features_intrinsic[selected_pts_mask],
                      "opacity": self._opacity[selected_pts_mask],
                      "scaling": self._scaling[selected_pts_mask],
                      "rotation": self._rotation[selected_pts_mask]}

        if self.use_kmap_pjmap or self.use_okmap:
            if self.use_kmap_pjmap and self.use_without_adaptive:
                new_tensor["box_coord"] = self.box_coord1[selected_pts_mask]
                new_tensor["box_coord2"] = self.box_coord2[selected_pts_mask]
            else:
                new_tensor["box_coord"] = self.box_coord[selected_pts_mask]

        self.densification_postfix(new_tensor)
        print(f"{selected_pts_mask.sum()} points cloned")

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, prune_more):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()

        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            if prune_more:
                prune_mask |= big_points_vs
                prune_mask |= big_points_ws
            else:
                prune_mask &= big_points_vs
                prune_mask &= big_points_ws
        self.prune_points(prune_mask)
        print(f"{prune_mask.sum()} points pruned")

        torch.cuda.empty_cache()

    def set_eval(self, eval=True):
        if eval:
            self.eval_mode = True

            if self.use_color_net:
                self.color_net = self.color_net.eval()
                self.color_net_origin_use_drop_out = self.color_net.use_drop_out
                self.color_net.use_drop_out = False

            if self.use_kmap_pjmap or self.use_okmap:
                self.origin_use_features_mask = self.map_generator.use_features_mask
                self.map_generator.use_features_mask = False
                self.use_features_mask = False
        else:
            self.eval_mode = False
            if self.use_color_net:
                self.color_net = self.color_net.train()
                self.color_net.use_drop_out = self.color_net_origin_use_drop_out
            if self.use_kmap_pjmap or self.use_okmap:
                self.use_features_mask = self.origin_use_features_mask
                self.map_generator.use_features_mask = self.origin_use_features_mask

    def add_densification_stats(self, viewspace_point_tensor, update_filter, width, height):
        grad = viewspace_point_tensor.grad.squeeze(0)  # [N, 2]
        # Normalize the gradient to [-1, 1] screen size
        grad[:, 0] *= width * 0.5
        grad[:, 1] *= height * 0.5
        self.xyz_gradient_accum[update_filter] += torch.norm(grad[update_filter, :2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def render(self, viewpoint_camera, pipe, bg_color: torch.Tensor, scaling_modifier=1.0, override_colors=None,
               other_viewpoint_camera=None, store_cache=False, use_cache=False, point_features=None,
               interp_alpha=0.0):
        """
        Render the scene.

        Background tensor (bg_color) must be on GPU!
        """
        if use_cache:
            self.forward_cache(viewpoint_camera)
        elif point_features is not None:
            self.forward_interpolate(viewpoint_camera, point_features)
        else:
            self.forward(viewpoint_camera, store_cache)

        if other_viewpoint_camera is not None:  # render using other camera center
            viewpoint_camera = other_viewpoint_camera

        means3D = self._xyz_dealt
        if pipe.deblur:
            gaussian_trans = viewpoint_camera.get_gaussian_trans(interp_alpha)
            means3D = F.pad(means3D, (0, 1), "constant", 1.0) @ gaussian_trans.T
        opacity = self.get_opacity_dealt  # [n_points, 1]
        scales = self.get_scaling * scaling_modifier
        rotations = self.get_rotation
        if self.use_colors_precomp:
            override_colors = self.get_colors  # [N, 3]
        if override_colors is None:
            colors = self._features_dealt  # [N, K, 3]
            sh_degree = self.active_sh_degree
        else:
            colors = override_colors
            sh_degree = None

        render_colors, render_alphas, info = rasterization(
            means=means3D,  # [N, 3]
            quats=rotations,  # [N, 4]
            scales=scales,  # [N, 3]
            opacities=opacity.squeeze(-1),  # [N,]
            colors=colors,
            viewmats=viewpoint_camera.world_view_transform[None],  # [1, 4, 4]
            Ks=viewpoint_camera.K[None],  # [1, 3, 3]
            backgrounds=bg_color[None],
            width=int(viewpoint_camera.image_width),
            height=int(viewpoint_camera.image_height),
            packed=False,
            sh_degree=sh_degree,
            render_mode="RGB+D" if pipe.use_depth_loss else "RGB",
            rasterize_mode="antialiased" if pipe.antialiasing else "classic",
        )

        # [1, H, W, 3] -> [3, H, W]
        rendered_image = render_colors[0].permute(2, 0, 1)
        radii = info["radii"].squeeze(0).max(dim=1).values  # .max(dim=1).values required if gsplat version >= 1.5
        viewspace_points = info["means2d"]
        if viewspace_points.requires_grad:  # in inference this will always be False
            viewspace_points.retain_grad()  # [1, N, 2]

        return {"render": rendered_image[:3],
                "viewspace_points": viewspace_points,
                "visibility_filter": radii > 0,
                "radii": radii,
                "depth": rendered_image[3] if pipe.use_depth_loss else None}
