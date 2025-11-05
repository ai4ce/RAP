import json

import wandb
from torch import optim, autocast, nn
from torch.amp import GradScaler
from torch.utils.data import DataLoader

from arguments import ModelParams, OptimizationParams, get_combined_args
from arguments.options import config_parser
from dataset_loaders.cambridge_scenes import Cambridge
from dataset_loaders.colmap_dataset import ColmapDataset
from dataset_loaders.seven_scenes import SevenScenes
from models.apr.discriminator import Discriminator
from models.apr.rapnet import RAPNet
from utils.early_stopper import EarlyStopper
from utils.eval_utils import eval_model
from utils.general_utils import fix_seed
from utils.model_utils import freeze_bn_layer_train, freeze_bn_layer
from utils.nt_xent_loss import NTXentLoss
from utils.nvs_utils import *
from utils.pose_utils import CameraPoseLoss
from utils.triplet_losses import TripletLossHardNegMiningPlus
from utils.vic_reg_loss import VICRegLoss

torch.set_float32_matmul_precision('high')


class BaseTrainer:
    def __init__(self, args):
        self.args = args
        if args.brisque_threshold:
            args.render_device = torch.device(args.render_device)
        # Create log dir and copy the config file
        self.logdir = f"{args.logbase}/{args.run_name}"
        os.makedirs(self.logdir, exist_ok=True)
        with open(f"{self.logdir}/args.txt", 'w') as file:
            for arg in sorted(vars(args)):
                attr = getattr(args, arg)
                file.write(f'{arg} = {attr}\n')
        if args.config is not None:
            with open(f"{self.logdir}/config.txt", 'w') as dst_file, open(args.config) as src_file:
                dst_file.write(src_file.read())
        self.full_ckpt_path = os.path.join(self.logdir, "full_checkpoint.pth")

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
        self.rap_hw = (rap_cam_params.h, rap_cam_params.w)
        gs_cam_params = CamParams(camera, args.resolution, args.render_device)
        gs_hw = (gs_cam_params.h, gs_cam_params.w)

        kwargs = dict(data_path=args.datadir, hw=self.rap_hw, hw_gs=gs_hw)
        train_set = dataset_class(train=True, train_skip=args.train_skip, **kwargs)
        val_set = dataset_class(train=False, test_skip=args.test_skip, **kwargs)
        train_dl = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=1)  # all data will be in the memory
        self.val_dl = DataLoader(val_set, batch_size=args.val_batch_size, shuffle=False, num_workers=args.val_num_workers)

        self.model = RAPNet(args).to(args.device)

        # freeze BN to not update gamma and beta
        if args.freeze_batch_norm:
            self.model = freeze_bn_layer(self.model)

        # set early stopper before compiling to enable inference on systems that do not support compiling
        self.early_stopper = EarlyStopper(self.logdir, self.model, patience=args.patience[0], verbose=False)

        if args.compile_model:
            self.model = torch.compile(self.model)

        # set optimizer
        self.optimizer_model = optim.Adam(self.model.parameters(), lr=args.learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_model, factor=0.95, patience=args.patience[1])
        self.schedulers = [self.scheduler]

        # loss functions
        self.pose_loss = CameraPoseLoss(args).to(args.device).train()
        self.mse_loss = nn.MSELoss(reduction='mean')

        # Initialize GradScaler
        self.scaler_model = GradScaler(args.device, enabled=args.amp)

        self.start_epoch = 0

        # load gaussian model
        if args.brisque_threshold:
            self.renderer = GaussianRendererWithBrisqueAttempts(args, gs_cam_params, self.rap_hw, train_dl, self.start_epoch)
        else:
            self.renderer = GaussianRendererWithAttempts(args, gs_cam_params, self.rap_hw, train_dl, self.start_epoch)
        if args.device != args.render_device:
            print(f"Training on device {args.device}. Rendering on device {args.render_device}. Starting renderer process.")
            self.renderer.start()
            self.poses, self.imgs_normed, self.imgs_rendered = self.renderer.queue.get()
        else:
            self.renderer.load_gaussians()
            self.poses, self.imgs_normed, self.imgs_rendered = self.renderer.render_set("train")
        self.dset_size = len(self.imgs_normed)
        self.n_iters = (self.dset_size + args.batch_size - 1) // args.batch_size

        self.run = wandb.init(project="RAP_NEW", config=args, name=args.run_name)

    def train_epoch(self, epoch, poses_perturbed, imgs_perturbed):
        """ we implement random view synthesis for generating more views to help training posenet """
        self.model.train()
        if self.args.freeze_batch_norm:
            self.model = freeze_bn_layer_train(self.model)

        train_loss_epoch = []

        # random generate batch_size of idx
        selected_indexes = np.random.choice(self.dset_size, size=[self.dset_size], replace=False)

        i_batch = 0
        batch_size = self.args.batch_size
        device = self.args.device
        loss_weights = self.args.loss_weights
        for _ in tqdm(range(self.n_iters), desc=f'Epoch {epoch}'):
            if i_batch + batch_size > self.dset_size:
                break
            batch_indexes = selected_indexes[i_batch:i_batch + batch_size]
            i_batch += batch_size

            imgs_normed_batch = self.imgs_normed[batch_indexes].to(device)
            poses_batch = self.poses[batch_indexes].reshape(batch_size, 12).to(device, torch.float)
            imgs_perturbed_batch = imgs_perturbed[batch_indexes].to(device) if imgs_perturbed is not None else None
            poses_perturbed_batch = poses_perturbed[batch_indexes].reshape(batch_size, 12).to(device, torch.float) if poses_perturbed is not None else None

            # Use autocast for mixed precision
            with autocast(device, enabled=self.args.amp, dtype=self.args.amp_dtype):
                # inference feature model for GT and nerf image
                _, poses_predicted = self.model(imgs_normed_batch, return_feature=False)
                loss_pose = self.pose_loss(poses_predicted, poses_batch)
                total_loss = loss_weights[0] * loss_pose

                if imgs_perturbed is not None and poses_perturbed is not None:
                    # inference model for RVS image
                    _, poses_perturbed_predicted = self.model(imgs_perturbed_batch, False)
                    loss_pose_perturbed = self.pose_loss(poses_perturbed_predicted, poses_perturbed_batch)
                    total_loss += loss_weights[2] * loss_pose_perturbed

            # Backward and optimization for Model
            self.optimizer_model.zero_grad(set_to_none=True)
            self.scaler_model.scale(total_loss).backward()
            self.scaler_model.step(self.optimizer_model)
            self.scaler_model.update()
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            train_loss_epoch.append(total_loss.item())
        train_loss = np.mean(train_loss_epoch)
        return train_loss

    def train(self):
        if self.args.resume:
            self.load_full_checkpoint(self.args.device)
        poses_perturbed, imgs_perturbed = None, None
        for epoch in tqdm(range(self.start_epoch, self.args.epochs), desc='Training'):
            do_rvs = epoch % self.args.rvs_refresh_rate == 0 or epoch == self.start_epoch  # decide if to re-synthesize new views
            if do_rvs:
                if self.args.device != self.args.render_device:
                    poses_perturbed, imgs_perturbed = self.renderer.queue.get()
                else:
                    self.renderer.epoch = epoch
                    poses_perturbed, imgs_perturbed = self.renderer.render_perturbed_imgs("train", self.poses)
            train_loss = self.train_epoch(epoch, poses_perturbed, imgs_perturbed)
            log_dict = eval_model(self.val_dl, self.model, self.mse_loss, self.args, vis=False)
            log_dict["Epoch"] = epoch
            log_dict["TrainLoss"] = train_loss
            for scheduler in self.schedulers:
                scheduler.step(log_dict["ValLoss"])
            # check whether to early stop
            self.early_stopper(log_dict, epoch=epoch, save_multiple=(not self.args.save_only_best), save_all=self.args.save_all_ckpt)
            tqdm.write(", ".join(f"{k}: {v}" for k, v in log_dict.items()))
            self.run.log(log_dict, step=epoch)
            if self.early_stopper.early_stop:
                print("Early stopped. Best result:")
                print(", ".join(f"{k}: {v}" for k, v in self.early_stopper.best_results.items()))
                self.run.log(self.early_stopper.best_results)
                break
            self.save_full_checkpoint(epoch, self.full_ckpt_path)
        if self.args.device != self.args.render_device:
            self.renderer.terminate()

    def save_full_checkpoint(self, epoch, save_path):
        state = self._save_full_checkpoint(epoch)
        torch.save(state, save_path)

    def _save_full_checkpoint(self, epoch):
        """
        Saves the full training state, allowing exact resume:
        - model, optimizers, schedulers
        - discriminator (if used)
        - EarlyStopper counters
        - current epoch
        """
        state = {
            'epoch': epoch,
            'model_state_dict': self.early_stopper.model.state_dict(),
            'optimizer_state_dict': self.optimizer_model.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'learnable_loss_state_dict': self.pose_loss.state_dict(),
            'early_stopper': self.early_stopper.state_dict()
        }
        return state

    def load_full_checkpoint(self, device='cuda'):
        if os.path.exists(self.full_ckpt_path):
            checkpoint = torch.load(self.full_ckpt_path, map_location=device, weights_only=False, mmap=True)
            self._load_full_checkpoint(checkpoint)
        else:
            print(f"No full checkpoint found at {self.full_ckpt_path}. Starting fresh.")

    def _load_full_checkpoint(self, checkpoint):
        """
        Loads the full state from disk if it exists.
        Returns the next epoch to continue training from.
        """
        self.early_stopper.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer_model.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.pose_loss.load_state_dict(checkpoint['learnable_loss_state_dict'])
        self.early_stopper.load_state_dict(checkpoint['early_stopper'])

        self.start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming training from epoch {self.start_epoch}")


class RVSTrainer(BaseTrainer):
    def __init__(self, args):
        super().__init__(args)
        if self.args.feature_loss == "triplet":
            self.feature_loss = TripletLossHardNegMiningPlus(self.args.triplet_margin)
        elif self.args.feature_loss == "vicreg":
            self.feature_loss = VICRegLoss().to(args.device).train()
        elif self.args.feature_loss == "ntxent":
            self.feature_loss = NTXentLoss()
        else:
            self.feature_loss = self.mse_loss

    def train_epoch(self, epoch, poses_perturbed, imgs_perturbed):
        """ we implement random view synthesis for generating more views to help training posenet """
        self.model.train()
        if self.args.freeze_batch_norm:
            self.model = freeze_bn_layer_train(self.model)

        train_loss_epoch = []

        # random generate batch_size of idx
        selected_indexes = np.random.choice(self.dset_size, size=[self.dset_size], replace=False)

        i_batch = 0
        batch_size = self.args.batch_size
        device = self.args.device
        loss_weights = self.args.loss_weights
        for _ in tqdm(range(self.n_iters), desc=f'Epoch {epoch}'):
            if i_batch + batch_size > self.dset_size:
                break
            batch_indexes = selected_indexes[i_batch:i_batch + batch_size]
            i_batch += batch_size

            imgs_normed_batch = self.imgs_normed[batch_indexes].to(device)
            imgs_rendered_batch = self.imgs_rendered[batch_indexes].to(device)
            poses_batch = self.poses[batch_indexes].reshape(batch_size, 12).to(device, torch.float)
            imgs_perturbed_batch = imgs_perturbed[batch_indexes].to(device)
            poses_perturbed_batch = poses_perturbed[batch_indexes].reshape(batch_size, 12).to(device, torch.float)

            # Use autocast for mixed precision
            with autocast(device, enabled=self.args.amp, dtype=self.args.amp_dtype):
                # inference feature model for GT and nerf image
                poses_batch = torch.cat((poses_batch, poses_batch))  # double gt pose tensor
                (features_target, features_rendered), poses_predicted = (
                    self.model(torch.cat((imgs_normed_batch, imgs_rendered_batch)), return_feature=True))

                loss_pose = self.pose_loss(poses_predicted, poses_batch)
                loss_feature = self.feature_loss(features_rendered, features_target)

                # inference model for RVS image
                _, poses_perturbed_predicted = self.model(imgs_perturbed_batch, False)

                loss_pose_perturbed = self.pose_loss(poses_perturbed_predicted, poses_perturbed_batch)

                total_loss = (loss_weights[0] * loss_pose +
                              loss_weights[1] * loss_feature +
                              loss_weights[2] * loss_pose_perturbed)

            # Backward and optimization for Model
            self.optimizer_model.zero_grad(set_to_none=True)
            self.scaler_model.scale(total_loss).backward()
            self.scaler_model.step(self.optimizer_model)
            self.scaler_model.update()
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            train_loss_epoch.append(total_loss.item())
        train_loss = np.mean(train_loss_epoch)
        return train_loss


class RVSWithDiscriminatorTrainer(RVSTrainer):
    def __init__(self, args):
        super().__init__(args)
        self.discriminator = Discriminator().to(args.device).train()
        if args.compile_model:
            self.discriminator = torch.compile(self.discriminator)
        self.optimizer_disc = optim.Adam(self.discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))
        self.scheduler_disc = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_disc, factor=0.95, patience=args.patience[1])
        self.schedulers.append(self.scheduler_disc)
        self.adversarial_loss = self.mse_loss
        self.scaler_disc = GradScaler(args.device, enabled=args.amp)
        size = (len(self.model.default_conf["hypercolumn_layers"]) * args.batch_size, 1)
        self.valid = torch.ones(size, device=args.device)
        self.fake = torch.zeros(size, device=args.device)

    def train_epoch(self, epoch, poses_perturbed, imgs_perturbed):
        """ we implement random view synthesis for generating more views to help training posenet """
        self.model.train()
        if args.freeze_batch_norm:
            self.model = freeze_bn_layer_train(self.model)

        train_loss_epoch = []

        # random generate batch_size of idx
        selected_indexes = np.random.choice(self.dset_size, size=[self.dset_size], replace=False)

        i_batch = 0
        batch_size = self.args.batch_size
        device = self.args.device
        loss_weights = self.args.loss_weights
        for _ in tqdm(range(self.n_iters), desc=f'Epoch {epoch}'):
            if i_batch + batch_size > self.dset_size:
                break
            batch_indexes = selected_indexes[i_batch:i_batch + batch_size]
            i_batch += batch_size

            imgs_normed_batch = self.imgs_normed[batch_indexes].to(device)
            imgs_rendered_batch = self.imgs_rendered[batch_indexes].to(device)
            poses_batch = self.poses[batch_indexes].reshape(batch_size, 12).to(device, torch.float)
            imgs_perturbed_batch = imgs_perturbed[batch_indexes].to(device)
            poses_perturbed_batch = poses_perturbed[batch_indexes].reshape(batch_size, 12).to(device, torch.float)

            # Use autocast for mixed precision
            with autocast(device, enabled=self.args.amp, dtype=self.args.amp_dtype):
                # inference feature model for GT and nerf image
                poses_batch = torch.cat((poses_batch, poses_batch))  # double gt pose tensor
                (features_target, features_rendered), poses_predicted = (
                    self.model(torch.cat((imgs_normed_batch, imgs_rendered_batch)), return_feature=True))

                # ---------------------
                #  Train Discriminator
                # ---------------------

                features_real = features_target.flatten(0, 1)
                features_fake = features_rendered.flatten(0, 1)

                disc_out_real = self.discriminator(features_real)
                disc_out_fake = self.discriminator(features_fake.detach())

                real_loss = self.adversarial_loss(disc_out_real, self.valid)
                fake_loss = self.adversarial_loss(disc_out_fake, self.fake)

                loss_disc = real_loss + fake_loss

            # Backward and optimization for Discriminator
            self.optimizer_disc.zero_grad(set_to_none=True)
            self.scaler_disc.scale(loss_disc).backward(retain_graph=True)
            self.scaler_disc.step(self.optimizer_disc)
            self.scaler_disc.update()
            # optimizer_disc.zero_grad()
            # loss_disc.backward(retain_graph=True)
            # optimizer_disc.step()

            with autocast(device, enabled=self.args.amp, dtype=self.args.amp_dtype):
                loss_pose = self.pose_loss(poses_predicted, poses_batch)
                loss_feature = self.feature_loss(features_rendered, features_target)

                # inference model for RVS image
                _, poses_perturbed_predicted = self.model(imgs_perturbed_batch, False)

                loss_pose_perturbed = self.pose_loss(poses_perturbed_predicted, poses_perturbed_batch)

                loss_generator = self.adversarial_loss(self.discriminator(features_fake), self.valid)

                total_loss = (loss_weights[0] * loss_pose +
                              loss_weights[1] * loss_feature +
                              loss_weights[2] * loss_pose_perturbed +
                              loss_weights[3] * loss_generator)

            # Backward and optimization for Model
            self.optimizer_model.zero_grad(set_to_none=True)
            self.scaler_model.scale(total_loss).backward()
            self.scaler_model.step(self.optimizer_model)
            self.scaler_model.update()
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            train_loss_epoch.append(total_loss.item())
        train_loss = np.mean(train_loss_epoch)
        return train_loss

    def _save_full_checkpoint(self, epoch):
        state = super()._save_full_checkpoint(epoch)
        state['discriminator_state_dict'] = self.discriminator.state_dict()
        state['optimizer_disc_state_dict'] = self.optimizer_disc.state_dict()
        state['scheduler_disc_state_dict'] = self.scheduler_disc.state_dict()
        return state

    def _load_full_checkpoint(self, checkpoint):
        super()._load_full_checkpoint(checkpoint)
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.optimizer_disc.load_state_dict(checkpoint['optimizer_disc_state_dict'])
        self.scheduler_disc.load_state_dict(checkpoint['scheduler_disc_state_dict'])


if __name__ == "__main__":
    parser = config_parser()
    model_params = ModelParams(parser)
    optimization = OptimizationParams(parser)
    args = get_combined_args(parser)
    model_params.extract(args)
    optimization.extract(args)
    fix_seed(args.seed)
    trainer = RVSWithDiscriminatorTrainer(args)
    trainer.train()
