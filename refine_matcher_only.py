import json
import math
import os.path

import wandb
from PIL import Image
from kornia.geometry import depth_to_3d_v2
from matplotlib import pyplot as plt
from torch import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from arguments import ModelParams, OptimizationParams, get_combined_args
from arguments.options import config_parser
from dataset_loaders.cambridge_scenes import Cambridge
from dataset_loaders.colmap_dataset import ColmapDataset
from dataset_loaders.seven_scenes import SevenScenes
from models.apr.rapnet import RAPNet
from refine import process_pose_file
from utils.cameras import CamParams
from utils.eval_utils import get_pose_error, vis_pose
from utils.general_utils import *
from utils.reverse_dfnet import reverse_poses

from matcher import Matcher
from dust3r_visloc.localization import run_pnp
# from vis3d import vis3d


torch.set_float32_matmul_precision('high')


@torch.no_grad()
def refine(args, run):
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
    imgs_rendered = sorted(os.listdir(f"{args.processed_root}/render"))
    imgs_depth = sorted(os.listdir(f"{args.processed_root}/depth_npy"))
    w, h = Image.open(f"{args.processed_root}/render/{imgs_rendered[0]}").size
    K = torch.tensor([[744/4, 0, w/2],
                      [0, 744/4, h/2],
                      [0, 0, 1]], device=args.device)

    kwargs = dict(data_path=args.datadir, hw=rap_hw, hw_gs=(h, w))
    val_set = dataset_class(train=False, test_skip=args.test_skip, **kwargs)
    test_dl = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=1)

    model = RAPNet(args=args).to(args.device).eval()
    if args.pretrained_model_path and os.path.exists(args.pretrained_model_path):
        print("Loading RAPNet from", args.pretrained_model_path)
        model.load_state_dict(torch.load(args.pretrained_model_path, map_location=args.device, mmap=True))
    else:
        print("Warning: No checkpoint found at", args.pretrained_model_path)
    if args.compile_model:
        model = torch.compile(model, mode="reduce-overhead")

    errors_trans = []
    errors_rot = []
    pred_trans = []

    matcher = Matcher(args.device, args.matcher_weights)

    if args.reprojection_error_diag_ratio > 0.0:
        reprojection_error_img = args.reprojection_error_diag_ratio * math.sqrt(w ** 2 + h ** 2)
    else:
        reprojection_error_img = args.reprojection_error

    x_coords = torch.arange(w, device=args.device).unsqueeze(0).expand(h, w)
    y_coords = torch.arange(h, device=args.device).unsqueeze(1).expand(h, w)
    upper_triangle_mask = y_coords < (x_coords * h / w)
    lower_triangle_mask = ~upper_triangle_mask

    font_size = round(58 * h / w)

    psnrs = np.empty(len(test_dl))
    times = np.empty(len(test_dl) - 1)

    if args.precomputed_dir:
        poses_pred = process_pose_file(f"{args.precomputed_dir}/poses.txt")
        pose_avg_stats = np.loadtxt(f"{args.datadir}/pose_avg_stats.txt")
        poses_pred = reverse_poses(poses_pred, pose_avg_stats)

    for i, (img_normed, pose, img_orig, img_name) in enumerate(tqdm(test_dl)):
        img_normed = img_normed.to(args.device)  # input
        pose = pose.reshape(3, 4)  # label
        img_orig = img_orig[0].to(args.device)
        img_name = img_name[0]
        reference = val_set.to_tensor(Image.open(f"{args.precomputed_dir}/render/{imgs_rendered[i]}")).to(args.device)
        depth = torch.tensor(np.load(f"{args.precomputed_dir}/depth_npy/{imgs_depth[i]}"), device=args.device)
        if args.precomputed_dir:
            ref_to_world_pred = poses_pred[i]
        else:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            with autocast(args.device, enabled=args.amp, dtype=args.amp_dtype):
                _, pose_pred = model(img_normed)
            end.record()
            torch.cuda.synchronize()

            pose_pred = pose_pred.reshape(3, 4).to("cpu", torch.double)
            ref_to_world_pred = np.eye(4)
            ref_to_world_pred[:3] = pose_pred.numpy()
        plt.close('all')
        fig, ax = plt.subplots(dpi=200)
        plt.subplots_adjust(top=1, bottom=0.2, left=0, right=1)
        # ax1 = plt.subplot2grid((3, 2), (0, 0), colspan=2)
        try:
            matches_im_query, matches_im_reference = matcher.match(img_orig, reference, args.confidence_threshold, ax)
            pos = ax.get_position()
            fig.text(0.5, pos.y0 - font_size / 700, f"{len(matches_im_query)} Matches", ha='center', fontsize=font_size / 2)
            plt.savefig(safe_path(f"{args.precomputed_dir}/match/{img_name}_before.jpg"), bbox_inches='tight', pad_inches=0, pil_kwargs={"quality": 100})

            y = np.rint(matches_im_reference[:, 1]).astype(int)
            x = np.rint(matches_im_reference[:, 0]).astype(int)
            pts3d = depth_to_3d_v2(depth, K, normalize_points=False).cpu().numpy()
            valid_pts3d = pts3d[y, x]
            # pts3d = depth_to_3d_v2(depth, intrinsic_matrix, normalize_points=False)[0]#.cpu().numpy()
            # valid_pts3d, matches_im_query, matches_im_map, matches_conf = matcher.match_coarse_to_fine(img_orig, reference, pts3d, intrinsic_matrix, args.pnp_max_points, args.confidence_threshold, ax1)
            success, query_to_ref = run_pnp(matches_im_query, valid_pts3d,
                                            K.cpu().numpy(), None,
                                            args.pnp_mode, reprojection_error_img, img_size=[w, h])  # 1.0
            if not success:
                raise ValueError("PnP failed")
            query_to_world = ref_to_world_pred @ query_to_ref
        except (IndexError, ValueError) as e:
            print(f"Failed to match image {i}: {e}")
            query_to_world = ref_to_world_pred
        if i > 0 and not args.precomputed_dir:
            elapsed = start.elapsed_time(end) / 1000
            # print(f"FPS: {1/elapsed:.2f}")
            times[i - 1] = 1/elapsed
        pred_trans.append(query_to_world[:3, 3])
        query_to_world = torch.tensor(query_to_world)
        dx_pred, dtheta_pred = get_pose_error(pose, torch.tensor(ref_to_world_pred))
        dx_refined, dtheta_refined = get_pose_error(pose, query_to_world)
        errors_trans.append(dx_refined)
        errors_rot.append(dtheta_refined)

        vis_before = img_orig.clone()
        vis_before[:, lower_triangle_mask] = reference[:, lower_triangle_mask].clamp(0, 1)
        plt.close('all')
        fig, ax = plt.subplots(dpi=200)
        plt.subplots_adjust(top=1, bottom=0.2, left=0, right=1)
        ax.imshow(vis_before.permute(1, 2, 0).cpu().numpy())
        ax.axis("off")
        line, = ax.plot([0, w - 1], [0, h - 1], linestyle='--', color='white', linewidth=2)
        line.set_dashes([5, 4])  # length, spacing
        pos = ax.get_position()
        fig.text(0.5, pos.y0 - font_size / 350, f"{dx_pred * 100:.2f} cm, {dtheta_pred:.2f}°", ha='center', fontsize=font_size)
        plt.savefig(safe_path(f"{args.precomputed_dir}/compare/{img_name}_before.jpg"), bbox_inches='tight', pad_inches=0, pil_kwargs={"quality": 100})

    mean_fps = np.mean(times)
    print(f"Average FPS: {mean_fps:.2f} FPS")

    median_psnr = np.median(psnrs)
    mean_psnr = np.mean(psnrs)
    max_psnr = np.max(psnrs)
    min_psnr = np.min(psnrs)
    print(f"Median PSNR: {median_psnr:.2f} dB")
    print(f"Mean PSNR: {mean_psnr:.2f} dB")
    print(f"Max PSNR: {max_psnr:.2f} dB")
    print(f"Min PSNR: {min_psnr:.2f} dB")

    errors_trans = np.hstack(errors_trans)
    errors_rot = np.hstack(errors_rot)
    pred_trans = np.vstack(pred_trans)
    gt_trans = val_set.poses.reshape((-1, 3, 4))[:, :3, 3]

    success_condition_5 = (errors_trans < 0.05) & (errors_rot < 5)
    success_condition_2 = (errors_trans < 0.02) & (errors_rot < 2)
    successful_count_5 = np.sum(success_condition_5)
    successful_count_2 = np.sum(success_condition_2)
    print(f"Successful count (5cm/5degree): {successful_count_5}")
    print(f"Successful count (2cm/2degree): {successful_count_2}")

    success_rate_5 = successful_count_5 / errors_trans.shape[0]
    success_rate_2 = successful_count_2 / errors_trans.shape[0]
    print(f"Success rate (5cm/5degree): {success_rate_5:.5%}")
    print(f"Success rate (2cm/2degree): {success_rate_2:.5%}")

    median_trans, median_rot = np.median(errors_trans), np.median(errors_rot)
    mean_trans, mean_rot = np.mean(errors_trans), np.mean(errors_rot)
    max_trans, max_rot = np.max(errors_trans), np.max(errors_rot)
    min_trans, min_rot = np.min(errors_trans), np.min(errors_rot)
    print(f'Median error: {median_trans} m, {median_rot} degrees.')
    print(f'Mean error: {mean_trans} m, {mean_rot} degrees.')
    print(f'Max error: {max_trans} m, {max_rot} degrees.')
    print(f'Min error: {min_trans} m, {min_rot} degrees.')

    run.log({"FPS": mean_fps, "AvgPSNR": mean_psnr, "MedPSNR": median_psnr, "MaxPSNR": max_psnr, "MinPSNR": min_psnr,
             "# Suc 5": successful_count_5, "# Suc 2": successful_count_2,
             "% Suc 5": success_rate_5, "% Suc 2": success_rate_2,
             "MedTrans": median_trans, "MedRot": median_rot, "AvgTrans": mean_trans, "AvgRot": mean_rot,
             "MaxTrans": max_trans, "MaxRot": max_rot, "MinTrans": min_trans, "MinRot": min_rot})

    vis_pose(pred_trans, gt_trans, errors_rot, args.precomputed_dir)


if __name__ == "__main__":
    parser = config_parser()
    model = ModelParams(parser, sentinel=True)
    optimization = OptimizationParams(parser)
    parser.add_argument("--confidence_threshold", type=float, default=-1.0,  # 1.001
                        help="confidence values higher than threshold are invalid")
    parser.add_argument("--reprojection_error", type=float, default=2.0, help="pnp reprojection error")
    parser.add_argument("--reprojection_error_diag_ratio", type=float, default=0.0,  # 0.08
                        help="pnp reprojection error as a ratio of the diagonal of the image")
    parser.add_argument("--pnp_max_points", type=int, default=100_000, help="pnp maximum number of points kept")
    parser.add_argument("--pnp_mode", type=str, default="cv2", choices=['cv2', 'poselib', 'pycolmap'],
                        help="pnp lib to use")
    parser.add_argument("--precomputed_dir", type=str, default="dfnet/shop", help="precomputed dir")
    args = get_combined_args(parser)
    fix_seed(args.seed)
    run = wandb.init(
        project="RAP",
        config=args,
        name=args.run_name
    )
    refine(args, run)
