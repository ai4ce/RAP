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

import json
import os
import random

from dataset_loaders.dataset_readers import sceneLoadTypeCallbacks
from models.gs.gaussian_model import GaussianModel
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from utils.general_utils import search_for_max_iteration


class Scene:
    def __init__(self, args, gaussians: GaussianModel, load_iteration=None, shuffle=True):
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.scene_name = args.scene_name
        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = search_for_max_iteration(os.path.join(self.model_path, "ckpts_point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print(f"Loading trained model at iteration {self.loaded_iter}")

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.depths, args.masks,
                                                          args.eval, args.use_depth_loss, args.depth_is_inverted,
                                                          args.use_masks, args.train_fraction)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval,
                                                           data_perturb=args.data_perturb)
        else:
            raise ValueError(f"No valid scene data found in source path {args.source_path}")

        if not self.loaded_iter:
            with (open(scene_info.ply_path, 'rb') as src_file,
                  open(os.path.join(self.model_path, "input.ply"), 'wb') as dest_file):
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            # random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        print("Loading Training Cameras")
        self.train_cameras = cameraList_from_camInfos(scene_info.train_cameras, args)
        print("Loading Test Cameras")
        self.test_cameras = cameraList_from_camInfos(scene_info.test_cameras, args)

        if self.loaded_iter:
            self.gaussians.load_ckpt_ply(os.path.join(self.model_path, "ckpts_point_cloud",
                                                      f"iteration_{self.loaded_iter}", "point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent, self.train_cameras[0].scale)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "ckpts_point_cloud", f"iteration_{iteration}")
        self.gaussians.save_ckpt_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
