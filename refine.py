import json
import math

import cv2 as cv
import wandb
from kornia.geometry import depth_to_3d_v2
from matplotlib import pyplot as plt
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from arguments import ModelParams, OptimizationParams, get_combined_args
from arguments.options import config_parser
from dataset_loaders.cambridge_scenes import Cambridge
from dataset_loaders.colmap_dataset import ColmapDataset
from dataset_loaders.seven_scenes import SevenScenes
from models.apr.rapnet import RAPNet
from models.gs.gaussian_model import GaussianModel
from utils.cameras import CamParams, Camera
from utils.eval_utils import vis_pose, get_pose_error
from utils.general_utils import *
from utils.image_utils import psnr
from utils.reverse_dfnet import reverse_poses

from matcher import Matcher
from dust3r_visloc.localization import run_pnp
# from vis3d import vis3d


torch.set_float32_matmul_precision('high')


def quaternion_to_rotation_matrix(qw, qx, qy, qz):
    R = np.array([
        [1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx ** 2 + qy ** 2)]
    ])
    return R


def pose_to_matrix(x, y, z, qw, qx, qy, qz):
    R = quaternion_to_rotation_matrix(qw, qx, qy, qz)
    pose = np.eye(4)
    pose[:3, :3] = R
    pose[:3, 3] = np.array([x, y, z])
    return pose


def process_pose_file(path):
    poses = []
    with open(path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith("#") or line == "":
                continue
            image_name, *data = line.split()
            qw, qx, qy, qz, x, y, z = map(float, data)
            pose = pose_to_matrix(x, y, z, qw, qx, qy, qz)
            poses.append((image_name, pose))
    poses.sort(key=lambda k: k[0])
    return np.array([pose for _, pose in poses])


def compute_transformation(origin_points, target_points):
    assert origin_points.shape == target_points.shape, "The two point clouds must have the same shape."

    origin_center = np.mean(origin_points, axis=0)
    target_center = np.mean(target_points, axis=0)

    centered_origin = origin_points - origin_center
    centered_target = target_points - target_center

    H = centered_origin.T @ centered_target
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = target_center - R @ origin_center
    H = np.eye(4)
    H[:3, :3] = R
    H[:3, 3] = t
    return H


class Refiner:
    @torch.no_grad()
    def __init__(self, args):
        if args.dataset_type == '7Scenes':
            dataset_class = SevenScenes
        elif args.dataset_type == 'Colmap':
            dataset_class = ColmapDataset
        elif args.dataset_type == 'Cambridge':
            dataset_class = Cambridge
        else:
            raise ValueError(f"Unknown dataset type: {args.dataset_type}")

        with open(f"{args.model_path}/cameras.json") as f:
            camera = json.load(f)[0]
        rap_cam_params = CamParams(camera, args.rap_resolution, args.device)
        rap_hw = (rap_cam_params.h, rap_cam_params.w)
        self.gs_cam_params = CamParams(camera, args.resolution, args.device)
        self.h, self.w = (self.gs_cam_params.h, self.gs_cam_params.w)

        kwargs = dict(data_path=args.datadir, hw=rap_hw, hw_gs=(self.h, self.w))
        val_set = dataset_class(train=False, test_skip=args.test_skip, **kwargs)
        self.test_dl = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=1)

        self.model = RAPNet(args=args).to(args.device).eval()
        if args.pretrained_model_path and os.path.exists(args.pretrained_model_path):
            print("Loading RAPNet from", args.pretrained_model_path)
            self.model.load_state_dict(torch.load(args.pretrained_model_path, map_location=args.device, mmap=True))
        else:
            print("Warning: No checkpoint found at", args.pretrained_model_path)
        if args.compile_model:
            self.model = torch.compile(self.model, mode="reduce-overhead")

        self.gaussians = GaussianModel(args)
        i = search_for_max_iteration(os.path.join(args.model_path, "ckpts_point_cloud"))
        self.gaussians.load_ckpt_ply(os.path.join(args.model_path, "ckpts_point_cloud", f"iteration_{i}", "point_cloud.ply"))
        bg_color = [1, 1, 1] if args.white_background else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float, device=args.device)
        self.gaussians.set_eval(True)
        args.deblur = False
        args.use_depth_loss = True

        self.errors_trans = []
        self.errors_rot = []
        self.pred_trans = []
        self.gt_trans = val_set.poses.reshape((-1, 3, 4))[:, :3, 3]

        self.matcher = Matcher(args.device, args.matcher_weights)

        if args.reprojection_error_diag_ratio > 0.0:
            self.reprojection_error_img = args.reprojection_error_diag_ratio * math.sqrt(self.w ** 2 + self.h ** 2)
        else:
            self.reprojection_error_img = args.reprojection_error

        if args.poses_txt:
            poses_pred = process_pose_file(args.poses_txt)
            pose_avg_stats = np.loadtxt(f"{args.datadir}/pose_avg_stats.txt")
            self.poses_pred = reverse_poses(poses_pred, pose_avg_stats)

    @torch.no_grad()
    def refine(self):
        for i, (img_normed, pose, img_orig, img_name) in enumerate(tqdm(self.test_dl)):
            img_normed = img_normed.to(args.device)  # input
            pose = pose.reshape(3, 4).double()  # label
            img_orig = img_orig[0].to(args.device)
            if args.poses_txt:
                ref_to_world_pred = self.poses_pred[i]
            else:
                with autocast(args.device, enabled=args.amp, dtype=args.amp_dtype):
                    _, pose_pred = self.model(img_normed)
                ref_to_world_pred = np.eye(4)
                ref_to_world_pred[:3] = pose_pred.reshape(3, 4).to("cpu", torch.double).numpy()
            world_to_ref_pred = np.linalg.inv(ref_to_world_pred)
            camera = Camera(None, None, None,
                            world_to_ref_pred[:3, :3], world_to_ref_pred[:3, 3], self.gs_cam_params.K,
                            self.gs_cam_params.FovX, self.gs_cam_params.FovY, img_orig)
            try:
                render_result = self.gaussians.render(camera, args, self.background)
                reference, depth = render_result['render'], render_result['depth']
                matches_im_query, matches_im_reference = self.matcher.match(img_orig, reference, args.confidence_threshold)
                y = np.rint(matches_im_reference[:, 1]).astype(int)
                x = np.rint(matches_im_reference[:, 0]).astype(int)
                pts3d = depth_to_3d_v2(depth, self.gs_cam_params.K, normalize_points=False).cpu().numpy()
                valid_pts3d = pts3d[y, x]
                success, query_to_ref = run_pnp(matches_im_query, valid_pts3d,
                                                self.gs_cam_params.K.cpu().numpy(), None,
                                                args.pnp_mode, self.reprojection_error_img, img_size=(self.w, self.h))  # 1.0
                if not success:
                    raise ValueError("PnP failed")
                query_to_world = ref_to_world_pred @ query_to_ref
            except (IndexError, ValueError) as e:
                print(f"Failed to match image {img_name}: {e}")
                query_to_world = ref_to_world_pred
            self.pred_trans.append(query_to_world[:3, 3])
            query_to_world = torch.tensor(query_to_world)
            dx_refined, dtheta_refined = get_pose_error(pose, query_to_world)
            self.errors_trans.append(dx_refined)
            self.errors_rot.append(dtheta_refined)

    @torch.no_grad()
    def report(self):
        self.errors_trans = np.hstack(self.errors_trans)
        self.errors_rot = np.hstack(self.errors_rot)
        self.pred_trans = np.vstack(self.pred_trans)

        success_condition_5 = (self.errors_trans < 0.05) & (self.errors_rot < 5)
        success_condition_2 = (self.errors_trans < 0.02) & (self.errors_rot < 2)
        successful_count_5 = np.sum(success_condition_5)
        successful_count_2 = np.sum(success_condition_2)
        print(f"Successful count (5cm/5degree): {successful_count_5}")
        print(f"Successful count (2cm/2degree): {successful_count_2}")

        success_rate_5 = successful_count_5 / self.errors_trans.shape[0]
        success_rate_2 = successful_count_2 / self.errors_trans.shape[0]
        print(f"Success rate (5cm/5degree): {success_rate_5:.5%}")
        print(f"Success rate (2cm/2degree): {success_rate_2:.5%}")

        median_trans, median_rot = np.median(self.errors_trans), np.median(self.errors_rot)
        mean_trans, mean_rot = np.mean(self.errors_trans), np.mean(self.errors_rot)
        max_trans, max_rot = np.max(self.errors_trans), np.max(self.errors_rot)
        min_trans, min_rot = np.min(self.errors_trans), np.min(self.errors_rot)
        print(f'Median error: {median_trans} m, {median_rot} degrees.')
        print(f'Mean error: {mean_trans} m, {mean_rot} degrees.')
        print(f'Max error: {max_trans} m, {max_rot} degrees.')
        print(f'Min error: {min_trans} m, {min_rot} degrees.')

        run = wandb.init(project="RAP", config=args, name=args.run_name)
        run.log({"# Suc 5": successful_count_5, "# Suc 2": successful_count_2,
                 "% Suc 5": success_rate_5, "% Suc 2": success_rate_2,
                 "MedTrans": median_trans, "MedRot": median_rot, "AvgTrans": mean_trans, "AvgRot": mean_rot,
                 "MaxTrans": max_trans, "MaxRot": max_rot, "MinTrans": min_trans, "MinRot": min_rot})
        return run


class VisualizationRefiner(Refiner):
    @torch.no_grad()
    def __init__(self, args):
        super().__init__(args)

        self.vis_root = f"vis/{args.run_name}"
        x_coords = torch.arange(self.w, device=args.device).unsqueeze(0).expand(self.h, self.w)
        y_coords = torch.arange(self.h, device=args.device).unsqueeze(1).expand(self.h, self.w)
        upper_triangle_mask = y_coords < (x_coords * self.h / self.w)
        self.lower_triangle_mask = ~upper_triangle_mask

        self.font_size = round(58 * self.h / self.w)

        self.psnrs = np.empty(len(self.test_dl))
        self.times = np.empty(len(self.test_dl) - 1)

    @torch.no_grad()
    def refine(self):
        start, end = None, None
        for i, (img_normed, pose, img_orig, img_name) in enumerate(tqdm(self.test_dl)):
            img_normed = img_normed.to(args.device)  # input
            pose = pose.reshape(3, 4).double()  # label
            img_orig = img_orig[0].to(args.device)
            img_name = img_name[0]
            gt_to_world = np.eye(4)
            gt_to_world[:3, :] = pose.numpy()
            world_to_gt = np.linalg.inv(gt_to_world)
            camera = Camera(None, None, None,
                            world_to_gt[:3, :3], world_to_gt[:3, 3], self.gs_cam_params.K,
                            self.gs_cam_params.FovX, self.gs_cam_params.FovY, img_orig)
            rendered = self.gaussians.render(camera, args, self.background)
            rendered, depth_rendered = rendered['render'].clamp(0, 1), rendered['depth']
            psnr_test = psnr(img_orig, rendered).double().mean().item()
            self.psnrs[i] = psnr_test
            plt.close('all')
            fig, ax = plt.subplots(dpi=200)
            plt.subplots_adjust(top=1, bottom=0.2, left=0, right=1)
            ax.imshow(rendered.permute(1, 2, 0).cpu().numpy())
            ax.axis("off")
            pos = ax.get_position()
            fig.text(0.5, pos.y0 - self.font_size / 350, f"{psnr_test:.2f} dB", ha='center', fontsize=self.font_size)
            plt.savefig(safe_path(f"{self.vis_root}/render/{img_name}_gt.jpg"), bbox_inches='tight', pad_inches=0,
                        pil_kwargs={"quality": 100})
            plt.imsave(safe_path(f"{self.vis_root}/depth/{img_name}_gt.jpg"), depth_rendered.cpu().numpy(),
                       cmap='plasma_r', pil_kwargs={"quality": 100})

            if args.poses_txt:
                ref_to_world_pred = self.poses_pred[i]
            else:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record(torch.cuda.current_stream())
                with autocast(args.device, enabled=args.amp, dtype=args.amp_dtype):
                    _, pose_pred = self.model(img_normed)
                end.record(torch.cuda.current_stream())
                torch.cuda.synchronize()

                ref_to_world_pred = np.eye(4)
                ref_to_world_pred[:3] = pose_pred.reshape(3, 4).to("cpu", torch.double).numpy()
            world_to_ref_pred = np.linalg.inv(ref_to_world_pred)
            camera = Camera(None, None, None,
                            world_to_ref_pred[:3, :3], world_to_ref_pred[:3, 3], self.gs_cam_params.K,
                            self.gs_cam_params.FovX, self.gs_cam_params.FovY, img_orig)
            render_result = self.gaussians.render(camera, args, self.background)
            reference, depth = render_result['render'], render_result['depth']
            ref = reference.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
            ref = cv.cvtColor(ref, cv.COLOR_RGB2BGR)
            cv.imwrite(safe_path(f"{self.vis_root}/render/{img_name}_before.jpg"), ref, [int(cv.IMWRITE_JPEG_QUALITY), 100])
            plt.imsave(safe_path(f"{self.vis_root}/depth/{img_name}_before.jpg"), depth.cpu().numpy(), cmap='plasma_r', pil_kwargs={"quality": 100})
            plt.close('all')
            fig, ax = plt.subplots(dpi=200)
            plt.subplots_adjust(top=1, bottom=0.2, left=0, right=1)
            # ax1 = plt.subplot2grid((3, 2), (0, 0), colspan=2)
            try:
                matches_im_query, matches_im_reference = self.matcher.match(img_orig, reference, args.confidence_threshold, ax)
                pos = ax.get_position()
                fig.text(0.5, pos.y0 - self.font_size / 700, f"{len(matches_im_query)} Matches", ha='center', fontsize=self.font_size / 2)
                plt.savefig(safe_path(f"{self.vis_root}/match/{img_name}_before.jpg"), bbox_inches='tight', pad_inches=0, pil_kwargs={"quality": 100})

                y = np.rint(matches_im_reference[:, 1]).astype(int)
                x = np.rint(matches_im_reference[:, 0]).astype(int)
                pts3d = depth_to_3d_v2(depth, self.gs_cam_params.K, normalize_points=False).cpu().numpy()
                valid_pts3d = pts3d[y, x]
                # pts3d = depth_to_3d_v2(depth, intrinsic_matrix, normalize_points=False)[0]#.cpu().numpy()
                # valid_pts3d, matches_im_query, matches_im_map, matches_conf = matcher.match_coarse_to_fine(img_orig, reference, pts3d, intrinsic_matrix, args.pnp_max_points, args.confidence_threshold, ax1)
                success, query_to_ref = run_pnp(matches_im_query, valid_pts3d,
                                                self.gs_cam_params.K.cpu().numpy(), None,
                                                args.pnp_mode, self.reprojection_error_img,
                                                img_size=(self.gs_cam_params.w, self.gs_cam_params.h))  # 1.0
                if not success:
                    raise ValueError("PnP failed")
                query_to_world = ref_to_world_pred @ query_to_ref
                world_to_query = np.linalg.inv(query_to_world)
                camera2 = Camera(None, None, None,
                                 world_to_query[:3, :3], world_to_query[:3, 3], self.gs_cam_params.K,
                                 self.gs_cam_params.FovX, self.gs_cam_params.FovY, img_orig)
                render_result2 = self.gaussians.render(camera2, args, self.background)
                reference2, depth2 = render_result2['render'], render_result2['depth']
                ref2 = reference2.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
                ref2 = cv.cvtColor(ref2, cv.COLOR_RGB2BGR)
                cv.imwrite(safe_path(f"{self.vis_root}/render/{img_name}_after.jpg"), ref2, [int(cv.IMWRITE_JPEG_QUALITY), 100])
                plt.imsave(safe_path(f"{self.vis_root}/depth/{img_name}_after.jpg"), depth2.cpu().numpy(), cmap='plasma_r', pil_kwargs={"quality": 100})
                # ax2 = plt.subplot2grid((3, 2), (1, 0), colspan=2)
                plt.close('all')
                fig, ax = plt.subplots(dpi=200)
                plt.subplots_adjust(top=1, bottom=0.2, left=0, right=1)
                matches_im_query2, matches_im_reference2 = self.matcher.match(img_orig, reference2, args.confidence_threshold, ax)
                pos = ax.get_position()
                fig.text(0.5, pos.y0 - self.font_size / 700, f"{len(matches_im_query2)} Matches", ha='center', fontsize=self.font_size / 2)
                plt.savefig(safe_path(f"{self.vis_root}/match/{img_name}_after.jpg"), bbox_inches='tight', pad_inches=0, pil_kwargs={"quality": 100})

                # plt.subplot(3, 2, 5)
                # plt.imshow(depth.cpu().numpy(), cmap='plasma_r')
                # plt.title("Pred Depth")
                # plt.axis("off")
                # plt.subplot(3, 2, 6)
                # plt.imshow(reference2.permute(1, 2, 0).cpu().numpy())
                # plt.title("Refined Image")
                # plt.axis("off")
                # plt.tight_layout()
                # plt.show()
                # plt.savefig(f"{vis_root}/test{img_name}.jpg")
            except (IndexError, ValueError) as e:
                print(f"Failed to match image {img_name}: {e}")
                query_to_world = ref_to_world_pred
                reference2 = reference
            if i > 0 and not args.poses_txt:
                elapsed = start.elapsed_time(end) / 1000
                # print(f"FPS: {1/elapsed:.2f}")
                self.times[i - 1] = 1/elapsed
            self.pred_trans.append(query_to_world[:3, 3])
            query_to_world = torch.tensor(query_to_world)
            dx_pred, dtheta_pred = get_pose_error(pose, torch.tensor(ref_to_world_pred))
            dx_refined, dtheta_refined = get_pose_error(pose, query_to_world)
            self.errors_trans.append(dx_refined)
            self.errors_rot.append(dtheta_refined)

            vis_before = img_orig.clone()
            vis_before[:, self.lower_triangle_mask] = reference[:, self.lower_triangle_mask].clamp(0, 1)
            plt.close('all')
            fig, ax = plt.subplots(dpi=200)
            plt.subplots_adjust(top=1, bottom=0.2, left=0, right=1)
            ax.imshow(vis_before.permute(1, 2, 0).cpu().numpy())
            ax.axis("off")
            line, = ax.plot([0, self.w - 1], [0, self.h - 1], linestyle='--', color='white', linewidth=2)
            line.set_dashes([5, 4])  # length, spacing
            pos = ax.get_position()
            fig.text(0.5, pos.y0 - self.font_size / 350, f"{dx_pred * 100:.2f} cm, {dtheta_pred:.2f}°", ha='center', fontsize=self.font_size)
            plt.savefig(safe_path(f"{self.vis_root}/compare/{img_name}_before.jpg"), bbox_inches='tight', pad_inches=0, pil_kwargs={"quality": 100})
            vis_after = img_orig.clone()
            vis_after[:, self.lower_triangle_mask] = reference2[:, self.lower_triangle_mask].clamp(0, 1)
            plt.close('all')
            fig, ax = plt.subplots(dpi=200)
            plt.subplots_adjust(top=1, bottom=0.2, left=0, right=1)
            ax.imshow(vis_after.permute(1, 2, 0).cpu().numpy())
            ax.axis("off")
            line, = ax.plot([0, self.w - 1], [0, self.h - 1], linestyle='--', color='white', linewidth=2)
            line.set_dashes([5, 4])
            pos = ax.get_position()
            fig.text(0.5, pos.y0 - self.font_size / 350, f"{dx_refined * 100:.2f} cm, {dtheta_refined:.2f}°", ha='center', fontsize=self.font_size)
            plt.savefig(safe_path(f"{self.vis_root}/compare/{img_name}_after.jpg"), bbox_inches='tight', pad_inches=0, pil_kwargs={"quality": 100})

    @torch.no_grad()
    def report(self):
        mean_fps = np.mean(self.times)
        print(f"Average FPS: {mean_fps:.2f} FPS")

        median_psnr = np.median(self.psnrs)
        mean_psnr = np.mean(self.psnrs)
        max_psnr = np.max(self.psnrs)
        min_psnr = np.min(self.psnrs)
        print(f"Median PSNR: {median_psnr:.2f} dB")
        print(f"Mean PSNR: {mean_psnr:.2f} dB")
        print(f"Max PSNR: {max_psnr:.2f} dB")
        print(f"Min PSNR: {min_psnr:.2f} dB")

        run = super().report()
        run.log({"FPS": mean_fps, "AvgPSNR": mean_psnr, "MedPSNR": median_psnr, "MaxPSNR": max_psnr, "MinPSNR": min_psnr}, step=0)

        vis_pose(self.pred_trans, self.gt_trans, self.errors_rot, self.vis_root)


if __name__ == "__main__":
    parser = config_parser()
    model_params = ModelParams(parser)
    optimization = OptimizationParams(parser)
    parser.add_argument("--matcher_weights", type=str, default="naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric",
                        help="path to the matcher weights")
    parser.add_argument("--confidence_threshold", type=float, default=-1.0,  # 1.001
                        help="confidence values higher than threshold are invalid")
    parser.add_argument("--reprojection_error", type=float, default=2.0, help="pnp reprojection error")
    parser.add_argument("--reprojection_error_diag_ratio", type=float, default=0.0,  # 0.08
                        help="pnp reprojection error as a ratio of the diagonal of the image")
    parser.add_argument("--pnp_max_points", type=int, default=100_000, help="pnp maximum number of points kept")
    parser.add_argument("--pnp_mode", type=str, default="cv2", choices=['cv2', 'poselib', 'pycolmap'],
                        help="pnp lib to use")
    parser.add_argument("--poses_txt", type=str, default=None, help="precomputed poses txt file")
    parser.add_argument("--cpu_affinity_ids", type=int, nargs="*", default=None, help="CPU affinity ID list in Python list format")
    args = get_combined_args(parser)
    args.device = args.render_device
    fix_seed(args.seed)

    if args.cpu_affinity_ids:
        import psutil
        psutil.Process().cpu_affinity(args.cpu_affinity_ids)

    refiner = Refiner(args)
    refiner.refine()
    refiner.report()
