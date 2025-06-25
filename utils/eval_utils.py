import math
import os

import numpy as np
import torch
from kornia.geometry import rotation_matrix_to_quaternion
from matplotlib import font_manager as fm
from matplotlib import gridspec
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch import autocast
from tqdm import tqdm

from utils.general_utils import safe_path


if "Times New Roman" in [f.name for f in fm.fontManager.ttflist]:
    plt.rcParams['font.family'] = 'Times New Roman'
elif "Times" in [f.name for f in fm.fontManager.ttflist]:
    plt.rcParams['font.family'] = 'Times'
else:
    print("Times New Roman font is not detected by matplotlib.")
    plt.rcParams['font.family'] = 'serif'


def vis_pose(pred_trans, gt_trans, errors_rot, save_path):
    """
    visualize predicted pose result vs. gt pose
    """
    ang_threshold = 10
    fig = plt.figure(figsize=(6, 7), dpi=200)
    gs = gridspec.GridSpec(2, 1, height_ratios=[16, 1])
    ax1 = fig.add_subplot(gs[0, 0], projection='3d')
    ax1.scatter(pred_trans[:, 0], pred_trans[:, 1], zs=pred_trans[:, 2], c='r', s=3 ** 2, label='Predicted')
    ax1.scatter(gt_trans[:, 0], gt_trans[:, 1], zs=gt_trans[:, 2], c='g', s=3 ** 2, label='Ground Truth')
    ax1.legend()
    ax1.view_init(30, 120)
    ax1.set_xlabel('$x$ (m)')
    ax1.set_ylabel('$y$ (m)')
    ax1.set_zlabel('$z$ (m)')

    # plot angular error
    ax2 = fig.add_subplot(gs[1, 0])
    seq_num = len(errors_rot)
    err = errors_rot.reshape(1, seq_num)
    err = np.tile(err, (20, 1))
    im = ax2.imshow(err, vmin=0, vmax=ang_threshold, aspect='auto')
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(im, cax=cax)
    ax2.set_yticks([])
    ax2.set_xticks([0, seq_num * 1 / 5, seq_num * 2 / 5, seq_num * 3 / 5, seq_num * 4 / 5, seq_num])
    plt.savefig(safe_path(save_path))


def get_pose_error(gt_pose, pred_pose):
    gt_q = rotation_matrix_to_quaternion(gt_pose[..., :3, :3])  # .cpu().numpy()
    gt_t = gt_pose[..., :3, 3]
    pred_q = rotation_matrix_to_quaternion(pred_pose[..., :3, :3])  # .cpu().numpy()
    pred_t = pred_pose[..., :3, 3]
    theta = gt_q.mul_(pred_q).sum(dim=-1).abs_().clamp_(-1., 1.).acos_().mul_(2 * 180 / math.pi)
    error_x = torch.linalg.vector_norm(gt_t - pred_t, ord=2, dim=-1)
    return error_x, theta

def generate_rotation_error_bar(errors_rot, output_dir):

    theta = errors_rot
    seq_num = len(theta)
    ang_threshold = 10

    os.makedirs(output_dir, exist_ok=True)
    
    tick_positions = [int(seq_num * frac) for frac in [1/5, 2/5, 3/5, 4/5, 1]]

    for i in tqdm(range(1, seq_num + 1), desc="Saving Error Bar"):
        fig, ax = plt.subplots(figsize=(18, 1))
        
        bar = np.full(seq_num, np.nan)
        bar[:i] = theta[:i]
        err = np.tile(bar.reshape(1, seq_num), (20, 1))
        # err = np.tile(theta[:i].reshape(1, i), (20, 1))
        
        im = ax.imshow(err, vmin=0, vmax=ang_threshold, aspect='auto')

        ax.set_yticks([])

        current_ticks = [pos for pos in tick_positions if pos <= i]
        if 0 not in current_ticks:
            current_ticks.insert(0, 0)
        ax.set_xticks(current_ticks)
        ax.set_xticklabels([str(pos) for pos in current_ticks], fontsize=35)

        ax.set_xlim(0, seq_num)

        frame_path = os.path.join(output_dir, f"frame_{i:03d}.jpg")
        plt.savefig(frame_path, bbox_inches='tight')
        plt.close(fig)

@torch.no_grad()
def eval_model(dl, model, loss, args, vis=True):
    """ Convert Rotation matrix to quaternion, then calculate the location errors. original from PoseNet Paper """
    model.eval()
    val_losses = []
    errors_trans = []
    errors_rot = []
    pred_trans = []
    directions = []
    for data, pose, _, _ in tqdm(dl, desc="Validating"):
        data = data.to(args.device)  # input
        pose = pose.to(args.device)  # label

        gt_pose = pose.float()
        with autocast(args.device, enabled=args.amp, dtype=args.amp_dtype):
            _, pred_pose = model(data)
            val_loss = loss(gt_pose, pred_pose)

        val_losses.append(val_loss.item())

        pose = pose.reshape((-1, 3, 4))
        pred_pose = pred_pose.reshape((-1, 3, 4)).double()

        # R_torch = pred_pose[:, :3, :3]
        # u, s, v = torch.svd(R_torch)
        # Rs = torch.matmul(u, v.transpose(-2, -1))
        # pred_pose[:, :3, :3] = Rs

        error_trans, error_rot = get_pose_error(pose, pred_pose)
        errors_trans.append(error_trans.cpu().numpy())
        errors_rot.append(error_rot.cpu().numpy())
        # print ('Iteration: {} Error XYZ (m): {} Error Q (degrees): {}'.format(i, error_x, theta))

        if vis:
            pred_trans.append(pred_pose[:, :3, 3].cpu().numpy())
            directions.append(pred_pose[:, :3, 2].cpu().numpy())

    mean_loss = np.mean(val_losses)

    errors_trans = np.hstack(errors_trans)
    errors_rot = np.hstack(errors_rot)

    median_trans, median_rot = np.median(errors_trans), np.median(errors_rot)
    mean_trans, mean_rot = np.mean(errors_trans), np.mean(errors_rot)
    max_trans, max_rot = np.max(errors_trans), np.max(errors_rot)
    min_trans, min_rot = np.min(errors_trans), np.min(errors_rot)

    success_condition_5 = (errors_trans < 0.05) & (errors_rot < 5)
    success_condition_2 = (errors_trans < 0.02) & (errors_rot < 2)
    successful_count_5 = np.sum(success_condition_5)
    successful_count_2 = np.sum(success_condition_2)
    success_rate_5 = successful_count_5 / errors_trans.shape[0]
    success_rate_2 = successful_count_2 / errors_trans.shape[0]

    save_dir = args.logbase
    save_name = f"{args.run_name}/"
    if args.pretrained_model_path:
        save_dir = os.path.dirname(args.pretrained_model_path) or "."
        save_name = f"{os.path.splitext(os.path.basename(args.pretrained_model_path))[0]}_"

    if vis:
        pred_trans = np.vstack(pred_trans)
        gt_trans = dl.dataset.poses.reshape((-1, 3, 4))[:, :3, 3]
        vis_pose(pred_trans, gt_trans, errors_rot, f"{save_dir}/{save_name}vis.jpg")

        norm = plt.Normalize(vmin=0, vmax=10)  # Adjust range as needed
        cmap = plt.get_cmap('cool')
        colors = []
        for error_trans in errors_trans:
            color = cmap(norm(error_trans))
            colors.append(color[:3])
        colors = np.array(colors)
        np.savez(safe_path(f"{save_dir}/{save_name}traj.npz"), points=pred_trans, colors=colors)
        generate_rotation_error_bar(errors_rot, output_dir=f"{save_dir}/{save_name}bar")
        directions = np.vstack(directions)
        with open(f"{save_dir}/{save_name}dir.txt", "w") as direction_file:
            for predicted_x, direction_vector in zip(pred_trans, directions):
                direction_file.write(f"{predicted_x[0]} {predicted_x[1]} {predicted_x[2]} {direction_vector[0]} {direction_vector[1]} {direction_vector[2]}\n")

    log_dict = {"Epoch": None, "TrainLoss": None, "ValLoss": mean_loss,
                "MedTrans": median_trans, "MedRot": median_rot,
                "AvgTrans": mean_trans, "AvgRot": mean_rot,
                "MaxTrans": max_trans, "MaxRot": max_rot,
                "MinTrans": min_trans, "MinRot": min_rot,
                "# Suc 5": successful_count_5, "# Suc 2": successful_count_2,
                "% Suc 5": success_rate_5, "% Suc 2": success_rate_2}

    return log_dict
