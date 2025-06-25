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
import logging
import os
from argparse import ArgumentParser
from pathlib import Path

import lpips
import torch
import torchvision.transforms.functional as tf
from fused_ssim import fused_ssim as ssim
from PIL import Image
from tqdm import tqdm

from utils.image_utils import psnr


def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())  #
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())  # 1,3,800,800
        image_names.append(fname)  #
    return renders, gts, image_names


def evaluate(model_paths, use_logs=False):
    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    lpips_alex = lpips.LPIPS(net='alex').to("cuda:0")
    for scene_dir in model_paths:
        try:
            print("Scene:", scene_dir)
            if use_logs:
                logging.info(f"Scene:{scene_dir}")
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}
            full_dict_polytopeonly[scene_dir] = {}
            per_view_dict_polytopeonly[scene_dir] = {}

            test_dir = Path(scene_dir) / "test"

            for method in os.listdir(test_dir):  #
                print("Method:", method)
                if use_logs:
                    logging.info(f"Method: {method}")
                full_dict[scene_dir][method] = {}  #
                per_view_dict[scene_dir][method] = {}
                full_dict_polytopeonly[scene_dir][method] = {}
                per_view_dict_polytopeonly[scene_dir][method] = {}

                method_dir = test_dir / method
                gt_dir = method_dir / "gt"  #
                renders_dir = method_dir / "renders"
                renders, gts, image_names = readImages(renders_dir, gt_dir)

                ssims = []
                psnrs = []
                lpipss = []

                for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                    _, C, H, W = renders[idx].shape
                    ssims.append(ssim(renders[idx][:, :, :, W // 2:], gts[idx][:, :, :, W // 2:]))
                    psnrs.append(psnr(renders[idx][:, :, :, W // 2:], gts[idx][:, :, :, W // 2:]))
                    lpipss.append(
                        lpips_alex(renders[idx][:, :, :, W // 2:], gts[idx][:, :, :, W // 2:], normalize=True))  # vgg

                print(f"  SSIM : {torch.tensor(ssims, dtype=torch.double).mean()}")
                print(f"  PSNR : {torch.tensor(psnrs, dtype=torch.double).mean()}")
                print(f"  LPIPS: {torch.tensor(lpipss, dtype=torch.double).mean()}")
                if use_logs:
                    logging.info(f"SSIM:{torch.tensor(ssims, dtype=torch.double).mean().item()}\n"
                                 f"PSNR:{torch.tensor(psnrs, dtype=torch.double).mean().item()}\n"
                                 f"LPIPS:{torch.tensor(lpipss, dtype=torch.double).mean().item()}")
                full_dict[scene_dir][method].update({"SSIM": torch.tensor(ssims, dtype=torch.double).mean().item(),
                                                     "PSNR": torch.tensor(psnrs, dtype=torch.double).mean().item(),
                                                     "LPIPS": torch.tensor(lpipss, dtype=torch.double).mean().item()})
                per_view_dict[scene_dir][method].update(
                    {"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                     "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                     "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)}})

            with open(scene_dir + "/results_half.json", 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
            with open(scene_dir + "/per_view_half.json", 'w') as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)
        except:
            print("Unable to compute metrics for model", scene_dir)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    args = parser.parse_args()
    evaluate(args.model_paths)
