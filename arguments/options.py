import configargparse
import torch


def eval_type(value):
    return eval(value)  # FIXME: not safe!!!


def config_parser():
    parser = configargparse.ArgumentParser()

    # General settings
    parser.add_argument("-c", "--config", is_config_file=True, help='Path to config file')
    parser.add_argument("-n", "--run_name", type=str, required=True, help='Experiment name')
    parser.add_argument("-l", "--logbase", type=str, default='logs', help='Base directory to store checkpoints and logs')
    parser.add_argument("-d", "--datadir", type=str, required=True, help='Input data directory')
    parser.add_argument("-p", "--pretrained_model_path", type=str, help='Path to pretrained model')

    # Dataset options
    parser.add_argument("-k", "--train_skip", type=int, default=1, help='Load 1/N images from training sets')
    parser.add_argument("--test_skip", type=int, default=1, help='Load 1/N images from test/validation sets')
    parser.add_argument("--sample_ratio", type=float, default=1, help='Sample ratio for test/val sets')
    parser.add_argument("-t", "--dataset_type", type=str, default='Colmap', help='Dataset type (Colmap/7Scenes/Cambridge)')
    parser.add_argument("--rap_resolution", type=float, default=2.0, help='Image downscale factor or resolution for RAP')

    # Training options
    parser.add_argument("--device", type=str, default='cuda', help='Device to run training on')
    parser.add_argument("--compile_model", action='store_true', default=True, help='Compile model to improve efficiency')
    parser.add_argument("--amp", action='store_true', default=True, help='Enable automatic mixed precision')
    parser.add_argument("--amp_dtype", type=eval_type, default='torch.float16', help='Mixed precision data type')
    parser.add_argument("--resume", action='store_true', default=False, help='Resume training from checkpoint')
    parser.add_argument("--epochs", type=int, default=2000, help='Max number of training epochs')
    parser.add_argument("--learning_rate", type=float, default=0.0001, help='Learning rate')
    parser.add_argument("--batch_size", type=int, default=8, help='Batch size for training')
    parser.add_argument("--val_batch_size", type=int, default=8, help='Batch size for validation')
    parser.add_argument("--val_num_workers", type=int, default=8, help='Number of workers for validation data loading')
    parser.add_argument("--feature_loss", type=str, default='vicreg', choices=['triplet', 'vicreg', 'ntxent', 'mse'], help='Feature loss type')
    parser.add_argument("--triplet_margin", type=float, default=1.0, help='Margin for triplet loss')
    parser.add_argument("--patience", type=int, nargs=2, default=[200, 50], help='EarlyStopping and reduceLR schedule')
    parser.add_argument("--freeze_batch_norm", action='store_true', default=True, help='Freeze BatchNorm layers during training')
    parser.add_argument("--seed", type=int, default=7, help='Random seed for (limited) reproducibility')
    parser.add_argument("--save_all_ckpt", action='store_true', default=False, help='Save all checkpoints for each epoch')
    parser.add_argument("--save_only_best", action='store_true', default=False, help='Save only best model')

    # Loss configuration
    parser.add_argument("--loss_weights", type=float, nargs=4, default=[1, 1, 1, 0.7], help='Weights for combined loss')
    parser.add_argument("--loss_learnable", action='store_true', default=True, help='Enable learnable pose loss')
    parser.add_argument("--loss_norm", type=int, default=2, help='Pose loss norm order')
    parser.add_argument("--s_x", type=float, default=-3, help='Pose loss s_x parameter')
    parser.add_argument("--s_q", type=float, default=-6.5, help='Pose loss s_q parameter')

    # RVS settings
    parser.add_argument("--max_attempts", type=int, default=100, help='Max attempts for RVS')
    parser.add_argument("--brisque_threshold", type=float, default=50, help='BRISQUE threshold for RVS')
    parser.add_argument("--no_appearance_augmentation", action='store_true', default=False, help='Disable appearance augmentation')
    parser.add_argument("--rvs_uniform_and_sphere", action='store_true', default=False, help='Use uniform and sphere sampling for RVS')
    parser.add_argument("--xz_plane_only", action='store_true', default=False, help='Use only XZ plane for RVS')
    parser.add_argument("--rvs_refresh_rate", type=int, default=2, help='Epochs between RVS refreshes')
    parser.add_argument("--rvs_trans", type=float, default=5, help='Translation jitter range for RVS')
    parser.add_argument("--rvs_rotation", type=float, default=1.2, help='Rotation jitter range for RVS (log_10 scale)')
    parser.add_argument("--d_max", type=float, default=1, help='Maximum RVS bound')

    # Visualization options
    parser.add_argument("--vis_rvs", action='store_true', default=False, help='Enable RVS visualization')
    parser.add_argument("--vis_featuremap", action='store_true', default=False, help='Visualize feature maps')
    parser.add_argument("--vis_channel", type=int, default=-1, help='Channel index for visualization')

    # Ablation studies
    parser.add_argument("--min_pose_distance", type=float, default=0, help='Minimum pose distance')
    parser.add_argument("--min_rotation_angle", type=float, default=0, help='Minimum rotation angle')

    # Model settings
    parser.add_argument("--learn_embedding_with_pose_token", action='store_true', default=True, help='Learn embedding with pose token')
    parser.add_argument("--reduction", type=str, nargs="*", default=["reduction_4", "reduction_3"], help='Reduction layers for feature extraction')
    parser.add_argument("--hidden_dim", type=int, default=256, help='Hidden dimension size')
    parser.add_argument("--num_heads", type=int, default=4, help='Number of attention heads')
    parser.add_argument("--feedforward_dim", type=int, default=256, help='Feedforward network dimension')
    parser.add_argument("--dropout", type=float, default=0.1, help='Dropout rate')
    parser.add_argument("--activation", type=str, default="gelu", choices=["relu", "gelu", "glu"], help='Activation function')
    parser.add_argument("--normalize_before", action='store_true', default=True, help='Normalize before feedforward layers')
    parser.add_argument("--num_encoder_layers", type=int, default=6, help='Number of encoder layers')

    return parser
