"""
pytorch data loader for the Cambridge Landmark dataset
"""
import os
import os.path as osp

import numpy as np
import torch
from torch.utils import data

from dataset_loaders.pose_reg_dataset import PoseRegDataset


class Cambridge(PoseRegDataset):
    @property
    def image_paths(self):
        return self.img_paths

    @property
    def image_names(self):
        return self.img_names

    @property
    def poses(self):
        return self._poses

    def __init__(self, data_path, train, hw, hw_gs, train_skip=1, test_skip=1, sample_ratio=1):
        """
        :param data_path: root 7scenes data directory.
        :param train: if True, return the training images. If False, returns the testing images
        :param train_skip: due to 7scenes are so big, now can use less training sets # of trainset = 1/train_skip
        :param test_skip: skip part of testset, # of testset = 1/test_skip
        :param hw: tuple of (height, width) to resize the images to
        """
        super().__init__(train, hw, hw_gs)

        if train:
            root_dir = osp.join(data_path, 'train')
        else:
            root_dir = osp.join(data_path, 'test')
        rgb_dir = osp.join(root_dir, 'rgb')
        pose_dir = osp.join(root_dir, 'poses')

        # collect poses and image names
        self.img_paths = []
        self.img_names = []
        for name in sorted(os.listdir(rgb_dir)):
            self.img_paths.append(osp.join(rgb_dir, name))
            self.img_names.append(osp.splitext(name)[0])
        self.pose_paths = [osp.join(pose_dir, f) for f in sorted(os.listdir(pose_dir))]

        # remove some abnormal data, need to fix later
        if 'shop' in data_path.lower() and self.train:
            del self.img_paths[42]
            del self.img_paths[35]
            del self.img_names[42]
            del self.img_names[35]
            del self.pose_paths[42]
            del self.pose_paths[35]

        if len(self.img_paths) != len(self.pose_paths):
            raise Exception('RGB file count does not match pose file count!')

        # train_skip and test_skip
        frame_idx = np.arange(len(self.img_paths))
        if train:
            if sample_ratio < 1:
                print("Downsample by ratio:", sample_ratio)
                num_samples = int(len(frame_idx) * sample_ratio)
                frame_idx = np.linspace(0, len(frame_idx) - 1, num_samples, dtype=int)
            elif train_skip > 1:
                print("Downsample by train_skip:", train_skip)
                frame_idx = frame_idx[::train_skip]
        elif test_skip > 1:
            frame_idx = frame_idx[::test_skip]

        self.img_paths = [self.img_paths[i] for i in frame_idx]
        self.img_names = [self.img_names[i] for i in frame_idx]
        self.pose_paths = [self.pose_paths[i] for i in frame_idx]

        # read poses
        self._poses = torch.tensor(np.array([np.loadtxt(pose)[:3].flatten() for pose in self.pose_paths]), dtype=torch.double)


def main():
    """
    visualizes the dataset
    """
    seq = 'ShopFacade'
    kwargs = dict(ret_hist=True)
    dset = Cambridge(seq, '../data/Cambridge/', True, train_skip=2, **kwargs)
    print('Loaded Cambridge sequence {:s}, length = {:d}'.format(seq, len(dset)))
    data_loader = data.DataLoader(dset, batch_size=4, shuffle=False)
    batch_count = 0
    N = 2
    for _ in data_loader:
        print('Minibatch {:d}'.format(batch_count))
        batch_count += 1
        if batch_count >= N:
            break


if __name__ == '__main__':
    main()
