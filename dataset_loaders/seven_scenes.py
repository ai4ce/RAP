"""
pytorch data loader for the 7-scenes dataset
"""
import os.path as osp
import glob

import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation
from torch.utils import data

from dataset_loaders.pose_reg_dataset import PoseRegDataset


def RT2QT(poses_in, mean_t, std_t):
    """
    processes the 1x12 raw pose from dataset by aligning and then normalizing
    :param poses_in: N x 12
    :param mean_t: 3
    :param std_t: 3
    :return: processed poses (translation + quaternion) N x 7
    """
    poses_out = np.zeros((len(poses_in), 7))
    poses_out[:, 0:3] = poses_in[:, [3, 7, 11]]

    # align
    for i in range(len(poses_out)):
        R = poses_in[i].reshape((3, 4))[:3, :3]
        q = Rotation.from_matrix(R).as_quat(canonical=True, scalar_first=True)
        poses_out[i, 3:] = q

    # normalize translation
    poses_out[:, :3] -= mean_t
    poses_out[:, :3] /= std_t
    return poses_out


def qlog(q):
    """
    Applies logarithm map to q
    :param q: (4,)
    :return: (3,)
    """
    if all(q[1:] == 0):
        q = np.zeros(3)
    else:
        q = np.arccos(q[0]) * q[1:] / np.linalg.norm(q[1:])
    return q


def process_poses_rotmat(poses_in, mean_t, std_t, align_R, align_t, align_s):
    """
    processes the 1x12 raw pose from dataset by aligning and then normalizing
    produce logq
    :param poses_in: N x 12
    :return: processed poses N x 12
    """
    return poses_in


def process_poses_q(poses_in, mean_t, std_t, align_R, align_t, align_s):
    """
    processes the 1x12 raw pose from dataset by aligning and then normalizing
    produce logq
    :param poses_in: N x 12
    :param mean_t: 3
    :param std_t: 3
    :param align_R: 3 x 3
    :param align_t: 3
    :param align_s: 1
    :return: processed poses (translation + log quaternion) N x 6
    """
    poses_out = np.zeros((len(poses_in), 6))  # (1000,6)
    poses_out[:, 0:3] = poses_in[:, [3, 7, 11]]  # x,y,z position
    # align
    for i in range(len(poses_out)):
        R = poses_in[i].reshape((3, 4))[:3, :3]  # rotation
        q = Rotation.from_matrix(align_R @ R).as_quat(canonical=True, scalar_first=True)
        poses_out[i, 3:] = q  # logq rotation
        t = poses_out[i, :3] - align_t
        poses_out[i, :3] = align_s * (align_R @ t[:, np.newaxis]).squeeze()

    # normalize translation
    poses_out[:, :3] -= mean_t  # (1000, 6)
    poses_out[:, :3] /= std_t
    return poses_out


def process_poses_logq(poses_in, mean_t, std_t, align_R, align_t, align_s):
    """
    processes the 1x12 raw pose from dataset by aligning and then normalizing
    produce logq
    :param poses_in: N x 12
    :param mean_t: 3
    :param std_t: 3
    :param align_R: 3 x 3
    :param align_t: 3
    :param align_s: 1
    :return: processed poses (translation + log quaternion) N x 6
    """
    poses_out = np.zeros((len(poses_in), 6))  # (1000,6)
    poses_out[:, 0:3] = poses_in[:, [3, 7, 11]]  # x,y,z position
    # align
    for i in range(len(poses_out)):
        R = poses_in[i].reshape((3, 4))[:3, :3]  # rotation
        q = Rotation.from_matrix(align_R @ R).as_quat(canonical=True, scalar_first=True)
        q = qlog(q)  # (1,3)
        poses_out[i, 3:] = q  # logq rotation
        t = poses_out[i, :3] - align_t
        poses_out[i, :3] = align_s * (align_R @ t[:, np.newaxis]).squeeze()

    # normalize translation
    poses_out[:, :3] -= mean_t  # (1000, 6)
    poses_out[:, :3] /= std_t
    return poses_out


def load_depth_image(filename):
    try:
        img_depth = Image.fromarray(np.array(Image.open(filename)).astype("uint16"))
    except IOError as e:
        print('Could not load image {:s}, IOError: {:s}'.format(filename, e))
        return None
    return img_depth


def normalize(x):
    return x / np.linalg.norm(x)


def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m


def normalize_recenter_pose(poses, sc, hwf):
    """ normalize xyz into [-1, 1], and recenter pose """
    target_pose = poses.reshape(poses.shape[0], 3, 4)
    target_pose[:, :3, 3] = target_pose[:, :3, 3] * sc

    x_norm = target_pose[:, 0, 3]
    y_norm = target_pose[:, 1, 3]
    z_norm = target_pose[:, 2, 3]

    tpose_ = target_pose + 0

    # find the center of pose
    center = np.array([x_norm.mean(), y_norm.mean(), z_norm.mean()])
    bottom = np.reshape([0, 0, 0, 1.], [1, 4])

    # pose avg
    vec2 = normalize(tpose_[:, :3, 2].sum(0))
    up = tpose_[:, :3, 1].sum(0)
    hwf = np.array(hwf).transpose()
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)
    c2w = np.concatenate([c2w[:3, :4], bottom], -2)

    bottom = np.tile(np.reshape(bottom, [1, 1, 4]), [tpose_.shape[0], 1, 1])
    poses = np.concatenate([tpose_[:, :3, :4], bottom], -2)
    poses = np.linalg.inv(c2w) @ poses
    return poses[:, :3, :].reshape(poses.shape[0], 12)


class SevenScenes(PoseRegDataset):
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
        """
        :param data_path: root 7scenes data directory.
        :param train: if True, return the training images. If False, returns the
        :param train_skip: due to 7scenes are so big, now can use less training sets # of trainset = 1/train_skip
        :param test_skip: skip part of testset, # of testset = 1/test_skip
        :param hw: (h, w) tuple, resize the image to this size
        """
        super().__init__(train, hw, hw_gs)

        # decide which sequences to use
        if train:
            split_file = osp.join(data_path, 'TrainSplit.txt')
        else:
            split_file = osp.join(data_path, 'TestSplit.txt')
        with open(split_file, 'r') as f:
            seqs = [int(l.split('sequence')[-1]) for l in f if not l.startswith('#')]  # parsing

        # read poses and collect image names
        self._image_paths = []
        self._image_names = []
        ps = {}
        vo_stats = {}
        for seq in seqs:
            seq_dir = f'{data_path}/seq-{seq:02d}'
            pose_paths = sorted(glob.glob(f'{seq_dir}/frame-*.pose.txt'))
            img_paths = sorted(glob.glob(f'{seq_dir}/frame-*.color.png'))

            # train_skip and test_skip
            if train and train_skip > 1:
                pose_paths = pose_paths[::train_skip]
                img_paths = img_paths[::train_skip]
            elif not train and test_skip > 1:
                pose_paths = pose_paths[::test_skip]
                img_paths = img_paths[::test_skip]

            ps[seq] = np.array([np.loadtxt(pose_path).flatten()[:12] for pose_path in pose_paths])  # list of all poses in file No. seq
            vo_stats[seq] = {'R': np.eye(3), 't': np.zeros(3), 's': 1}
            self._image_paths += img_paths
            self._image_names += [f'seq-{seq:02d}/{osp.splitext(osp.basename(p))[0]}' for p in img_paths]

        pose_stats_filename = osp.join(data_path, 'pose_stats.txt')
        if train:
            mean_t = np.zeros(3)  # optionally, use the ps dictionary to calc stats
            std_t = np.ones(3)
            np.savetxt(pose_stats_filename, np.vstack((mean_t, std_t)), fmt='%8.7f')
        else:
            mean_t, std_t = np.loadtxt(pose_stats_filename)

        # convert pose to translation + log quaternion
        logq = False
        quat = False
        poses = []
        for seq in seqs:
            if logq:
                pss = process_poses_logq(poses_in=ps[seq], mean_t=mean_t, std_t=std_t, align_R=vo_stats[seq]['R'],
                                         align_t=vo_stats[seq]['t'],
                                         align_s=vo_stats[seq]['s'])  # here returns t + logQed R
            elif quat:
                pss = RT2QT(poses_in=ps[seq], mean_t=mean_t, std_t=std_t)  # here returns t + quaternion R
            else:
                pss = process_poses_rotmat(poses_in=ps[seq], mean_t=mean_t, std_t=std_t, align_R=vo_stats[seq]['R'],
                                           align_t=vo_stats[seq]['t'], align_s=vo_stats[seq]['s'])
            poses.append(pss)
        self._poses = torch.tensor(np.vstack(poses), dtype=torch.double)


def main():
    """
    visualizes the dataset
    """
    # from common.vis_utils import show_batch, show_stereo_batch
    import torchvision.transforms as transforms
    seq = 'heads'
    num_workers = 6
    transform = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dset = SevenScenes(seq, '../data/deepslam_data/7Scenes', True, transform)
    print('Loaded 7Scenes sequence {:s}, length = {:d}'.format(seq, len(dset)))
    pdb.set_trace()

    data_loader = data.DataLoader(dset, batch_size=4, shuffle=True, num_workers=num_workers)

    batch_count = 0
    N = 2
    for batch in data_loader:
        print('Minibatch {:d}'.format(batch_count))
        pdb.set_trace()
        # if mode < 2:
        #   show_batch(make_grid(batch[0], nrow=1, padding=25, normalize=True))
        # elif mode == 2:
        #   lb = make_grid(batch[0][0], nrow=1, padding=25, normalize=True)
        #   rb = make_grid(batch[0][1], nrow=1, padding=25, normalize=True)
        #   show_stereo_batch(lb, rb)

        batch_count += 1
        if batch_count >= N:
            break


if __name__ == '__main__':
    main()
