import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
import random

from visloc import fine_matching, resize_image_to_max, crop

from mast3r.fast_nn import fast_reciprocal_NNs
from mast3r.model import AsymmetricMASt3R
from mast3r.utils.coarse_to_fine import select_pairs_of_crops
from dust3r_visloc.datasets.utils import get_resize_function, get_HW_resolution, rescale_points3d
from dust3r.inference import inference
from dust3r.utils.image import convert_images
from dust3r.utils.geometry import geotrf


class Matcher:
    image_mean = torch.tensor([0.5, 0.5, 0.5]).reshape(1, 3, 1, 1)
    image_std = torch.tensor([0.5, 0.5, 0.5]).reshape(1, 3, 1, 1)

    def __init__(self, device, model_name=None):
        self.device = device
        self.fast_nn_params = dict(device=device, dist='dot', block_size=2**13)
        self.model = AsymmetricMASt3R.from_pretrained(model_name or "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric").to(device).eval()
        self.maxdim = max(self.model.patch_embed.img_size)
        self.cmap = plt.get_cmap('jet')

    def match_coarse_to_fine(self, query, reference, pts3d, intrinsics, max_match=100000, conf_thr=-1, vis_fig=None):
        coarse_matches_im0, coarse_matches_im1 = self.match(query, reference, conf_thr)

        query_pil = Image.fromarray(query.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy())
        query_rgb_tensor, query_K, query_to_orig_max, query_to_resize_max, (HQ, WQ) = resize_image_to_max(
            None, query_pil, intrinsics)
        reference_pil = Image.fromarray(reference.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy())
        map_rgb_tensor, map_K, map_to_orig_max, map_to_resize_max, (HM, WM) = resize_image_to_max(
            None, reference_pil, intrinsics)
        WM_full, HM_full = reference_pil.size
        valid_all = torch.ones((HM_full, WM_full), dtype=torch.bool, device=self.device)
        if WM_full != WM or HM_full != HM:
            height, width, _ = pts3d.shape
            y_full, x_full = torch.meshgrid(torch.arange(height), torch.arange(width))
            pos2d_cv2 = torch.stack([x_full, y_full], dim=-1).numpy().astype(np.float64)
            _, _, pts3d_max, valid_max = rescale_points3d(pos2d_cv2, pts3d, map_to_resize_max, HM, WM)
            pts3d = torch.from_numpy(pts3d_max)
            valid_all = torch.from_numpy(valid_max)

        coarse_matches_im0 = geotrf(query_to_resize_max, coarse_matches_im0, norm=True)
        coarse_matches_im1 = geotrf(map_to_resize_max, coarse_matches_im1, norm=True)

        crops1, crops2 = [], []
        crops_v1, crops_p1 = [], []
        to_orig1, to_orig2 = [], []
        map_resolution = get_HW_resolution(HM, WM, maxdim=self.maxdim, patchsize=self.model.patch_embed.patch_size)
        query_resolution = get_HW_resolution(HQ, WQ, maxdim=self.maxdim, patchsize=self.model.patch_embed.patch_size)
        for crop_q, crop_b, pair_tag in select_pairs_of_crops(map_rgb_tensor,
                                                              query_rgb_tensor,
                                                              coarse_matches_im1,
                                                              coarse_matches_im0,
                                                              maxdim=self.maxdim,
                                                              overlap=.5,
                                                              forced_resolution=[map_resolution,
                                                                                 query_resolution]):
            # Per crop processing
            c1, v1, p1, trf1 = crop(map_rgb_tensor, valid_all, pts3d, crop_q, None)
            c2, _, _, trf2 = crop(query_rgb_tensor, None, None, crop_b, None)
            crops1.append(c1)
            crops2.append(c2)
            crops_v1.append(v1)
            crops_p1.append(p1)
            to_orig1.append(trf1)
            to_orig2.append(trf2)

        if len(crops1) == 0 or len(crops2) == 0:
            valid_pts3d, matches_im_query, matches_im_map, matches_conf = [], [], [], []
        else:
            crops1, crops2 = torch.stack(crops1), torch.stack(crops2)
            if len(crops1.shape) == 3:
                crops1, crops2 = crops1[None], crops2[None]
            crops_v1 = torch.stack(crops_v1)
            crops_p1 = torch.stack(crops_p1)
            to_orig1, to_orig2 = torch.stack(to_orig1), torch.stack(to_orig2)
            map_crop_view = dict(img=crops1.permute(0, 3, 1, 2),
                                 instance=['1' for _ in range(crops1.shape[0])],
                                 valid=crops_v1, pts3d=crops_p1,
                                 to_orig=to_orig1)
            query_crop_view = dict(img=crops2.permute(0, 3, 1, 2),
                                   instance=['2' for _ in range(crops2.shape[0])],
                                   to_orig=to_orig2)

            # Inference and Matching
            valid_pts3d, matches_im_query, matches_im_map, matches_conf = fine_matching(query_crop_view,
                                                                                        map_crop_view,
                                                                                        self.model, self.device,
                                                                                        48,
                                                                                        5,
                                                                                        self.fast_nn_params)
            matches_im_query = geotrf(query_to_orig_max, matches_im_query, norm=True)
            matches_im_map = geotrf(map_to_orig_max, matches_im_map, norm=True)

            if conf_thr >= 0:
                mask = matches_conf >= conf_thr
                valid_pts3d = valid_pts3d[mask]
                matches_im_query = matches_im_query[mask]
                matches_im_map = matches_im_map[mask]
                matches_conf = matches_conf[mask]

            if len(matches_im_query) > max_match:
                idxs = random.sample(range(len(matches_im_query)), max_match)
                valid_pts3d = valid_pts3d[idxs]
                matches_im_query = matches_im_query[idxs]

        if vis_fig is not None:
            self.visualize_matches({"img": query_rgb_tensor.permute(2, 0, 1)},
                                   {"img": map_rgb_tensor.permute(2, 0, 1)}, matches_im_query, matches_im_map, vis_fig)

        print(f"Number of fine matches: {matches_im_query.shape[0]}")
        return valid_pts3d, matches_im_query, matches_im_map, matches_conf

    def match(self, query, reference, conf_thr=-1, vis_fig=None):
        if query.shape != reference.shape:
            raise ValueError("Two images need to have the same shape.")

        orig_h, orig_w = query.shape[1], query.shape[2]
        images = convert_images([query, reference], size=512)
        output = inference([tuple(images)], self.model, self.device, batch_size=1, verbose=False)

        # at this stage, you have the raw dust3r predictions
        view1, pred1 = output['view1'], output['pred1']
        view2, pred2 = output['view2'], output['pred2']
        conf1, conf2 = pred1['desc_conf'].squeeze(0).cpu().numpy(), pred2['desc_conf'].squeeze(0).cpu().numpy()
        desc1, desc2 = pred1['desc'].squeeze(0).detach(), pred2['desc'].squeeze(0).detach()

        # find 2D-2D matches between the two images
        matches_im0, matches_im1 = fast_reciprocal_NNs(desc1, desc2, subsample_or_initxy1=8,
                                                       device=self.device, dist='dot', block_size=2**13)

        # ignore small border around the edge
        new_h, new_w = view1['true_shape'][0]
        new_h = int(new_h)
        new_w = int(new_w)
        valid_matches_im0 = (matches_im0[:, 0] >= 3) & (matches_im0[:, 0] < new_w - 3) & (
            matches_im0[:, 1] >= 3) & (matches_im0[:, 1] < new_h - 3)

        valid_matches_im1 = (matches_im1[:, 0] >= 3) & (matches_im1[:, 0] < new_w - 3) & (
            matches_im1[:, 1] >= 3) & (matches_im1[:, 1] < new_h - 3)

        valid_matches = valid_matches_im0 & valid_matches_im1
        matches_im0, matches_im1 = matches_im0[valid_matches], matches_im1[valid_matches]

        if conf_thr >= 0:
            matches_confs = np.minimum(
                conf2[matches_im1[:, 1], matches_im1[:, 0]],
                conf1[matches_im0[:, 1], matches_im0[:, 0]]
            )
            valid_matches = matches_confs >= conf_thr
            matches_im0, matches_im1 = matches_im0[valid_matches], matches_im1[valid_matches]

        if vis_fig is not None:
            self.visualize_matches(view1, view2, matches_im0, matches_im1, vis_fig)

        resize_func, to_resize, to_orig_1 = get_resize_function(self.maxdim,
                                                              self.model.patch_embed.patch_size,
                                                              orig_h, orig_w)

        resize_func, to_resize, to_orig_2 = get_resize_function(self.maxdim,
                                                              self.model.patch_embed.patch_size,
                                                              orig_h, orig_w)

        matches_im_query = matches_im0.astype(np.float64)
        matches_im_map = matches_im1.astype(np.float64)

        # if orig_h == new_h:
        #     if orig_w != new_w:
        #         matches_im_query[:, 0] += (orig_w - new_w) / 2
        #         matches_im_map[:, 0] += (orig_w - new_w) / 2
        # else:
        #     if orig_w == new_w:
        #         matches_im_query[:, 1] += (orig_h - new_h) / 2
        #         matches_im_map[:, 1] += (orig_h - new_h) / 2
        #     else:
        #         raise NotImplementedError("Only center cropping is supported. Scaling is not supported.")

        matches_im_query[:, 0] += 0.5
        matches_im_query[:, 1] += 0.5
        matches_im_map[:, 0] += 0.5
        matches_im_map[:, 1] += 0.5
        matches_im_query = geotrf(to_orig_1, matches_im_query, norm=True)
        matches_im_map = geotrf(to_orig_2, matches_im_map, norm=True)
        matches_im_query[:, 0] -= 0.5
        matches_im_query[:, 1] -= 0.5
        matches_im_map[:, 0] -= 0.5
        matches_im_map[:, 1] -= 0.5

        print(f"Number of coarse matches: {matches_im_query.shape[0]}")
        return matches_im_query, matches_im_map

    def visualize_matches(self, view1, view2, matches_im0, matches_im1, vis_fig):
        """visualize a few matches"""
        n_viz = 20
        num_matches = matches_im0.shape[0]
        match_idx_to_viz = np.round(np.linspace(0, num_matches - 1, n_viz)).astype(int)
        viz_matches_im0, viz_matches_im1 = matches_im0[match_idx_to_viz], matches_im1[match_idx_to_viz]

        viz_imgs = []
        for i, view in enumerate([view1, view2]):
            rgb_tensor = view['img'] * self.image_std + self.image_mean
            viz_imgs.append(rgb_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy())

        H0, W0, H1, W1 = *viz_imgs[0].shape[:2], *viz_imgs[1].shape[:2]
        img0 = np.pad(viz_imgs[0], ((0, max(H1 - H0, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
        img1 = np.pad(viz_imgs[1], ((0, max(H0 - H1, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
        img = np.concatenate((img0, img1), axis=1)
        vis_fig.imshow(img)
        for i in range(n_viz):
            (x0, y0), (x1, y1) = viz_matches_im0[i].T, viz_matches_im1[i].T
            vis_fig.plot([x0, x1 + W0], [y0, y1], '-+', color=self.cmap(i / (n_viz - 1)), scalex=False, scaley=False)
        vis_fig.axis('off')
        # vis_fig.set_title('Left: GT Image, Right: Pred Image')
        # plt.tight_layout()
        # plt.show()
