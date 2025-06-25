import os

import numpy as np
import torch

from dataset_loaders.dataset_readers import readColmapSceneInfo
from dataset_loaders.pose_reg_dataset import PoseRegDataset


class ColmapDataset(PoseRegDataset):
    @property
    def image_paths(self):
        return self._image_paths

    @property
    def image_names(self):
        return self._image_names

    @property
    def poses(self):
        return self._poses

    def __init__(self, data_path, train, hw, hw_gs, train_skip=1, test_skip=1):
        super().__init__(train, hw, hw_gs)

        scene_info = readColmapSceneInfo(data_path, "images", eval=True)
        # n_train_cams = len(scene_info.train_cameras)
        # n_test_cams = len(scene_info.test_cameras)
        # n_cams = n_train_cams + n_test_cams
        if train:
            cameras = scene_info.train_cameras
            # print(f"Train cameras: {n_train_cams} of {n_cams} ({n_train_cams / n_cams:.2%})")
        else:
            cameras = scene_info.test_cameras
            # print(f"Test cameras: {n_test_cams} of {n_cams} ({n_test_cams / n_cams:.2%})")

        self._image_paths = []
        self._image_names = []
        poses = []
        i = 0
        for cam in cameras:
            i += 1
            if train and i % train_skip != 0:
                continue
            if not train and i % test_skip != 0:
                continue
            self._image_paths.append(cam.image_path)
            self._image_names.append(os.path.splitext(cam.image_name)[0])
            w2c = np.eye(4)
            w2c[:3, :3] = cam.R
            w2c[:3, 3] = cam.T
            pose = np.linalg.inv(w2c)
            poses.append(pose[:3].flatten())
        self._poses = torch.tensor(np.array(poses), dtype=torch.double)
