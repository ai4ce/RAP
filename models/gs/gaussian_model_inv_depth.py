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
import math

import torch
from torch.nn import functional as F
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

from models.gs.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh


class GaussianModelInvDepth(GaussianModel):
    def add_densification_stats(self, viewspace_point_tensor, update_filter, *args):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter, :2], dim=-1,
                                                             keepdim=True)
        self.denom[update_filter] += 1

    def render(self, viewpoint_camera, pipe, bg_color: torch.Tensor, scaling_modifier=1.0, override_colors=None,
               other_viewpoint_camera=None, store_cache=False, use_cache=False, point_features=None, interp_alpha=0.0):
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

        # Set up rasterization configuration
        raster_settings = GaussianRasterizationSettings(
            image_height=viewpoint_camera.image_height,
            image_width=viewpoint_camera.image_width,
            tanfovx=math.tan(viewpoint_camera.FoVx * 0.5),
            tanfovy=math.tan(viewpoint_camera.FoVy * 0.5),
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform.T,  # 4*4
            projmatrix=viewpoint_camera.full_proj_transform.T,  # 4*4
            sh_degree=self.active_sh_degree,  # 0,
            campos=viewpoint_camera.camera_center,  # 3
            prefiltered=False,
            debug=pipe.debug,
            antialiasing=pipe.antialiasing
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = torch.zeros_like(self._xyz, dtype=torch.float, requires_grad=True,
                                              device=self.device) + 0  # [n_points, 3]
        if screenspace_points.requires_grad:  # in inference this will always be False
            screenspace_points.retain_grad()

        means3D = self._xyz_dealt
        if pipe.deblur:
            gaussian_trans = viewpoint_camera.get_gaussian_trans(interp_alpha)
            means3D = F.pad(means3D, (0, 1), "constant", 1.0) @ gaussian_trans.T
        means2D = screenspace_points
        opacity = self.get_opacity_dealt  # [n_points, 1]

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = None
        if pipe.compute_cov3D_python:
            cov3D_precomp = self.get_covariance(scaling_modifier)
        else:
            scales = self.get_scaling  # [n_points, 3]
            rotations = self.get_rotation  # [n_points, 4]

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = None
        if not pipe.convert_SHs_python and self.use_colors_precomp:
            override_colors = self.get_colors
        if override_colors is None:
            if pipe.convert_SHs_python:
                shs_view = self._features_dealt.transpose(1, 2).view(-1, 3, (self.max_sh_degree + 1) ** 2)  # N, 3, 16
                # dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))   #N, 3
                # dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
                dir_pp_normalized = self.view_direction
                sh2rgb = eval_sh(self.active_sh_degree, shs_view, dir_pp_normalized)
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            else:
                shs = self._features_dealt
        else:
            colors_precomp = override_colors

        # Rasterize visible Gaussians to image, obtain their radii (on screen).
        rendered_image, radii, rendered_depth = rasterizer(
            means3D=means3D,  # [n_points, 3]
            means2D=means2D,  # [n_points, 3]
            shs=shs,  # [n_points, 16, 3]
            colors_precomp=colors_precomp,
            opacities=opacity,  # [n_points, 1]
            scales=scales,  # [n_points, 3]
            rotations=rotations,  # [n_points, 4]
            cov3D_precomp=cov3D_precomp)

        return {"render": rendered_image,
                "viewspace_points": screenspace_points,
                "visibility_filter": radii > 0,
                "radii": radii,
                "depth": rendered_depth}
