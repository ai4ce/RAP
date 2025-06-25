import numpy as np


def normalize(v):
    """Normalize a vector."""
    return v / np.linalg.norm(v)


def average_poses(poses):
    """
    Calculate the average pose, which is then used to center all poses
    using @center_poses. Its computation is as follows:
    1. Compute the center: the average of pose centers.
    2. Compute the z axis: the normalized average z axis.
    3. Compute axis y': the average y axis.
    4. Compute x' = y' cross product z, then normalize it as the x axis.
    5. Compute the y axis: z cross product x.
    Note that at step 3, we cannot directly use y' as y axis since it's
    not necessarily orthogonal to z axis. We need to pass from x to y.
    Inputs:
        poses: (N_images, 4, 4)
    Outputs:
        pose_avg: (3, 4) the average pose
    """
    poses = poses[:, :3, :]
    # 1. Compute the center
    center = poses[..., 3].mean(0)  # (3)
    # 2. Compute the z axis
    z = normalize(poses[..., 2].mean(0))  # (3)
    # 3. Compute axis y' (no need to normalize as it's not the final output)
    y_ = poses[..., 1].mean(0)  # (3)
    # 4. Compute the x axis
    x = normalize(np.cross(y_, z))  # (3)
    # 5. Compute the y axis (as z and x are normalized, y is already of norm 1)
    y = np.cross(z, x)  # (3)
    pose_avg = np.stack([x, y, z, center], 1)  # (3, 4)
    return pose_avg


def reverse_poses(all_poses_homo, pose_avg_from_file=None):
    """
    Reverse the transformations applied to the poses and reconstruct the original poses.

    Inputs:
        all_poses: (N_images, 4, 4) the transformed poses.
        pose_avg_from_file: (3, 4) optional average pose matrix, loaded from a file if provided.

    Outputs:
        original_poses: (N_images, 4, 4) the reconstructed original poses.
        pose_avg: (3, 4) the average pose used during the centering process.
    """

    def rot_phi(phi):
        """Generate a rotation matrix for rotation around the X-axis."""
        return np.array([
            [1, 0, 0, 0],
            [0, np.cos(phi), -np.sin(phi), 0],
            [0, np.sin(phi), np.cos(phi), 0],
            [0, 0, 0, 1]
        ]).astype(float)

    def reverse_center_poses(poses_homo, pose_avg_from_file=None):
        """

        Inputs:
            poses: (N_images, 4, 4)
            pose_avg_from_file: if not None, pose_avg is loaded from pose_avg_stats.txt

        Outputs:
            poses_centered: (N_images, 4, 4) the centered poses
            pose_avg: (3, 4) the average pose
        """

        if pose_avg_from_file is None:
            pose_avg = average_poses(poses_homo)  # (3, 4) # this need to be fixed throughout dataset
        else:
            pose_avg = pose_avg_from_file  # (3, 4)

        pose_avg_homo = np.eye(4)  # (4, 4)
        pose_avg_homo[:3] = pose_avg  # convert to homogeneous coordinate for faster computation (4,4)
        poses_centered = pose_avg_homo @ poses_homo  # (N_images, 4, 4)
        return poses_centered, pose_avg  # np.linalg.inv(pose_avg_homo)

    # 2. Reverse the X-axis mirror transformation
    mirror_matrix = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
    all_poses_homo[:, :3, :3] = all_poses_homo[:, :3, :3] @ mirror_matrix

    # 3. Reverse the view direction correction
    all_poses_homo[:, :3, :3] = -all_poses_homo[:, :3, :3]

    # 4. Reverse the 180-degree X-axis rotation
    all_poses_homo = rot_phi(-180 / 180. * np.pi) @ all_poses_homo

    # 5. Reverse the centering process
    all_poses_original, pose_avg = reverse_center_poses(all_poses_homo, pose_avg_from_file)

    return all_poses_original
