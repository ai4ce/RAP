import json

import torch
# import intel_extension_for_pytorch as ipex
from torch import nn
from torch.utils.data import DataLoader

import wandb
from arguments import ModelParams, OptimizationParams, get_combined_args
from arguments.options import config_parser
from dataset_loaders.cambridge_scenes import Cambridge
from dataset_loaders.colmap_dataset import ColmapDataset
from dataset_loaders.seven_scenes import SevenScenes
from models.apr.rapnet import RAPNet
from utils.cameras import CamParams
from utils.eval_utils import eval_model
from utils.general_utils import fix_seed
from utils.model_utils import vis_featuremap

torch.set_float32_matmul_precision("high")


if __name__ == '__main__':
    parser = config_parser()
    model = ModelParams(parser)
    optimization = OptimizationParams(parser)
    args = get_combined_args(parser)
    model.extract(args)
    optimization.extract(args)
    fix_seed(args.seed)

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
    # args.device = "xpu" if torch.xpu.is_available() else "cpu"
    # args.amp = False
    rap_cam_params = CamParams(camera, args.rap_resolution, args.device)
    rap_hw = (rap_cam_params.h, rap_cam_params.w)
    gs_cam_params = CamParams(camera, args.resolution, args.device)
    gs_hw = (gs_cam_params.h, gs_cam_params.w)

    kwargs = dict(data_path=args.datadir, hw=rap_hw, hw_gs=gs_hw)
    val_set = dataset_class(train=False, test_skip=args.test_skip, **kwargs)
    val_dl = DataLoader(val_set, batch_size=args.val_batch_size, shuffle=False, num_workers=args.val_num_workers)

    model = RAPNet(args=args).to(args.device)

    print("Load RAPNet from", args.pretrained_model_path)
    # ckpt = torch.load(args.pretrained_model_path, map_location=args.device, mmap=True, weights_only=False)
    # model.load_state_dict(ckpt['model_state_dict'])
    model.load_state_dict(torch.load(args.pretrained_model_path, map_location=args.device, mmap=True))
    model.eval()
    # model = ipex.optimize(model, weights_prepack=False)
    # model = torch.compile(model, backend="ipex")
    if args.vis_featuremap:
        vis_featuremap(args, val_dl, model, rap_hw)
    mse_loss = nn.MSELoss(reduction='mean')
    log_dict = eval_model(val_dl, model, mse_loss, args)
    print("\n".join(f"{k}: {v}" for k, v in log_dict.items()))

    wandb.init(project="RAP_A6000", config=args, name=args.run_name).log(log_dict)
