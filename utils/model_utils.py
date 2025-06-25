"""
helper functions to train robust feature extractors
"""

import matplotlib.pyplot as plt
import torch
from torch import nn, autocast
from torch.nn import functional as F
from torchvision.utils import make_grid
from tqdm import tqdm

from utils.general_utils import safe_path


def save_feature_gs(args, idx, feature_target, hw):
    for i, feature in enumerate(feature_target):
        feature_target_path = f"vis/feature_map/{args.run_name}/{i}/{idx:05d}.jpg"
        feature = feature[:, args.vis_channel, :, :]
        feature = F.interpolate(feature[None], size=hw, mode='bilinear', align_corners=False).squeeze()
        plt.imsave(safe_path(feature_target_path), feature.cpu().numpy(), cmap='jet_r', pil_kwargs={"quality": 100})


@torch.no_grad()
def vis_featuremap(args, dl, model, hw):
    """ visualize feature map """
    model.eval()
    for i, (data, pose, _, _) in enumerate(tqdm(dl, desc="Visualizing Feature Map")):
        data = data.to(args.device)
        inputs = data.to(args.device)
        with autocast(args.device, enabled=args.amp, dtype=args.amp_dtype):
            feature_list, predicted_pose = model(inputs, return_feature=True, is_single_stream=True)
        feature_target = feature_list[0]
        save_feature_gs(args, i, feature_target, hw)


def freeze_bn_layer(model):
    """ freeze bn layer by not require grad but still behave differently when model.train() vs. model.eval() """
    print("Freezing BatchNorm Layers...")
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            # print("this is a BN layer:", module)
            if hasattr(module, 'weight'):
                module.weight.requires_grad_(False)
            if hasattr(module, 'bias'):
                module.bias.requires_grad_(False)
    return model


def freeze_bn_layer_train(model):
    """ set batchnorm to eval()
        it is useful to align train and testing result
    """
    # model.train()
    # print("Freezing BatchNorm Layers...")
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()
    return model


def save_image_saliancy(tensor, path, normalize: bool = False, scale_each: bool = False, ):
    """
    Modification based on TORCHVISION.UTILS
    ::param: tensor (batch, channel, H, W)
    """
    # grid = make_grid(tensor.detach(), normalize=normalize, scale_each=scale_each, nrow=32)
    grid = make_grid(tensor.detach(), normalize=normalize, scale_each=scale_each, nrow=6)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    fig = plt.figure()
    plt.imshow(ndarr[:, :, 0], cmap='jet')  # viridis, plasma
    plt.axis('off')
    fig.savefig(path, bbox_inches='tight', dpi=fig.dpi, pad_inches=0.0)
    plt.close()


def save_image_saliancy_single(tensor, path, normalize: bool = False, scale_each: bool = False, ):
    """
    Modification based on TORCHVISION.UTILS, save single feature map
    ::param: tensor (batch, channel, H, W)
    """
    grid = make_grid(tensor.detach(), normalize=normalize, scale_each=scale_each, nrow=1)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    fig = plt.figure()
    # plt.imshow(ndarr[:,:,0], cmap='plasma') # viridis, jet
    plt.imshow(ndarr[:, :, 0], cmap='jet')  # viridis, jet
    plt.axis('off')
    fig.savefig(path, bbox_inches='tight', dpi=fig.dpi, pad_inches=0.0)
    plt.close()


def print_feature_examples(features, path):
    """
    print feature maps
    ::param: features
    """
    for i in range(len(features)):
        fn = path + '{}.png'.format(i)
        save_image_saliancy(features[i].permute(1, 0, 2, 3), fn, normalize=True)


def plot_features(features, path='f', isList=True):
    """
    print feature maps
    :param features: (3, [batch, H, W]) or [3, batch, H, W]
    :param path: save image path
    :param isList: wether the features is an list
    :return:
    """
    kwargs = {'normalize': True, }  # 'scale_each' : True

    if isList:
        dim = features[0].dim()
    else:
        dim = features.dim()
    assert (dim == 3 or dim == 4)

    if dim == 4 and isList:
        print_feature_examples(features, path)
    elif dim == 4 and (isList == False):
        fn = path
        lvl, b, H, W = features.shape
        for i in range(features.shape[0]):
            fn = path + '{}.png'.format(i)
            save_image_saliancy(features[i][None, ...].permute(1, 0, 2, 3).cpu(), fn, normalize=True)

        # # concat everything
        # features = features.reshape([-1, H, W])
        # # save_image(features[None,...].permute(1,0,2,3).cpu(), fn, **kwargs)
        # save_image_saliancy(features[None,...].permute(1,0,2,3).cpu(), fn, normalize=True) 

    elif dim == 3 and isList:  # print all images in the list
        for i in range(len(features)):
            fn = path + '{}.png'.format(i)
            # save_image(features[i][None,...].permute(1,0,2,3).cpu(), fn, **kwargs)
            save_image_saliancy(features[i][None, ...].permute(1, 0, 2, 3).cpu(), fn, normalize=True)
    elif dim == 3 and (isList == False):
        fn = path
        save_image_saliancy(features[None, ...].permute(1, 0, 2, 3).cpu(), fn, normalize=True)
