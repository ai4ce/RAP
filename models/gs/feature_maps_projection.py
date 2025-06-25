import torch
from torch.nn import functional as F


def project2d(pointcloud, world2camera, camera2image, box_coord, feature_map):
    xyz_world_h = F.pad(pointcloud, (0, 1), "constant", 1)  # N,4
    # xyz_camera = (xyz_world_h @ world2camera.T)[:, :3]  # N,3
    # xy_image0 = xyz_camera @ camera2image.T  # N,3
    xy_image0 = xyz_world_h @ (world2camera.T[:, :3] @ camera2image.T)
    xy_image = torch.zeros((xy_image0.shape[0], 2), device=xy_image0.device)
    xy_image[:, 0] = xy_image0[:, 0] / (xy_image0[:, 2] + 1e-2)
    xy_image[:, 1] = xy_image0[:, 1] / (xy_image0[:, 2] + 1e-2)  # N，3

    # mask_in_width
    valid_point_mask = (0 < xy_image[:, 0])
    valid_point_mask &= (xy_image[:, 0] < box_coord[:, 1])  # 3,H,W
    # mask_in_height
    valid_point_mask &= (0 < xy_image[:, 1])
    valid_point_mask &= (xy_image[:, 1] < box_coord[:, 0])
    # mask_in_image = mask_in_width & mask_in_height
    # mask_front_point
    valid_point_mask &= (xy_image0[:, 2] > 0)
    # valid_point_mask = mask_in_image & mask_front_point

    valid_pixel = xy_image[valid_point_mask][:, :2]
    if box_coord.shape[0] > 1:
        valid_box_coord = box_coord[valid_point_mask]
    else:
        valid_box_coord = box_coord
    valid_pixelx_normalized = valid_pixel[:, 0] / (valid_box_coord[:, 1] / 2) - 1
    valid_pixely_normalized = valid_pixel[:, 1] / (valid_box_coord[:, 0] / 2) - 1
    valid_pixel_normal = torch.stack((valid_pixelx_normalized, valid_pixely_normalized), dim=1)
    valid_pixel_normal = torch.unsqueeze(valid_pixel_normal, 0)
    valid_pixel_normal = torch.unsqueeze(valid_pixel_normal, 0)
    point_feature = F.grid_sample(feature_map, valid_pixel_normal, mode='bilinear',
                                  padding_mode='border', align_corners=False).squeeze().T
    #print("pointcloud shape:", pointcloud.shape)
    #print("point_feature shape:", point_feature.shape)
    point_feature_all = torch.zeros(size=(pointcloud.shape[0], point_feature.shape[1]),
                                    dtype=pointcloud.dtype, device=pointcloud.device)

    point_feature_all[valid_point_mask] = point_feature

    return point_feature_all, valid_point_mask


if __name__ == "__main__":
    pass
