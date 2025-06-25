import numpy as np
import torch
from tqdm import tqdm

from utils.general_utils import safe_path


class EarlyStopper:
    """Early stops the training if validation criteria doesn't improve after a given patience."""

    # source https://blog.csdn.net/qq_37430422/article/details/103638681
    def __init__(self, logdir, model, patience=50, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 50
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.early_stop = False
        # self.val_loss_min = np.inf
        self.trans_err_min = np.inf
        self.rot_err_min = np.inf
        self.delta = delta
        self.model = model
        self.ckpt_save_dir = logdir
        self.best_results = None

    def __call__(self, log_dict, epoch=-1, save_multiple=False, save_all=False):
        trans_err = log_dict["MedTrans"]
        rot_err = log_dict["MedRot"]
        if trans_err < self.trans_err_min and rot_err < self.rot_err_min:
            self.counter = 0
            if self.verbose:
                tqdm.write('Saving model:\n'
                           # f'Validation loss: {self.val_loss_min:.6f} --> {val_loss:.6f}'
                           f'Translation error: {self.trans_err_min:.6f} --> {trans_err:.6f}\n'
                           f'Rotation error: {self.rot_err_min:.6f} --> {rot_err:.6f}')
            # self.val_loss_min = val_loss
            self.trans_err_min = trans_err
            self.rot_err_min = rot_err
            self.best_results = log_dict
            if save_multiple:
                best_ckpt_name = f'ckpt-{epoch:04d}-{trans_err:.6f}-{rot_err:.6f}.pt'
            else:
                best_ckpt_name = f'ckpt-best.pt'
            torch.save(self.model.state_dict(), safe_path(f'{self.ckpt_save_dir}/{best_ckpt_name}'))
        else:
            self.counter += 1
            if self.verbose:
                tqdm.write(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
            if save_all:  # save all ckpt
                torch.save(self.model.state_dict(), safe_path(f'{self.ckpt_save_dir}/ckpt-{trans_err:.6f}-{rot_err:.6f}.pt'))

    def is_best_model(self):
        """ Check if current model the best one.
        get early stop counter, if counter==0: it means current model has the best validation loss
        """
        return self.counter == 0

    def state_dict(self):
        return {
            'counter': self.counter,
            'best_results': self.best_results,
            # 'val_loss_min': self.val_loss_min,
            'trans_err_min': self.trans_err_min,
            'rot_err_min': self.rot_err_min,
        }

    def load_state_dict(self, state_dict):
        self.counter = state_dict['counter']
        self.best_results = state_dict['best_results']
        # self.val_loss_min = state_dict['val_loss_min']
        self.trans_err_min = state_dict['trans_err_min']
        self.rot_err_min = state_dict['rot_err_min']
