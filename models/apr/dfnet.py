import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models

from models.apr.rapnet import AdaptLayers2

# VGG-16 Layer Names and Channels
vgg16_layers = {
    "conv1_1": 64,
    "relu1_1": 64,
    "conv1_2": 64,
    "relu1_2": 64,
    "pool1": 64,
    "conv2_1": 128,
    "relu2_1": 128,
    "conv2_2": 128,
    "relu2_2": 128,
    "pool2": 128,
    "conv3_1": 256,
    "relu3_1": 256,
    "conv3_2": 256,
    "relu3_2": 256,
    "conv3_3": 256,
    "relu3_3": 256,
    "pool3": 256,
    "conv4_1": 512,
    "relu4_1": 512,
    "conv4_2": 512,
    "relu4_2": 512,
    "conv4_3": 512,
    "relu4_3": 512,
    "pool4": 512,
    "conv5_1": 512,
    "relu5_1": 512,
    "conv5_2": 512,
    "relu5_2": 512,
    "conv5_3": 512,
    "relu5_3": 512,
    "pool5": 512,
}


class DFNet(nn.Module):
    """ DFNet implementation """
    default_conf = {
        'hypercolumn_layers': ["conv1_2", "conv3_3", "conv5_3"],
        'output_dim': 128,
    }

    def __init__(self, feat_dim=12):
        super(DFNet, self).__init__()

        self.layer_to_index = {k: v for v, k in enumerate(vgg16_layers.keys())}
        self.hypercolumn_indices = [self.layer_to_index[n] for n in
                                    self.default_conf['hypercolumn_layers']]  # [2, 14, 28]

        # Initialize architecture
        vgg16 = models.vgg16(pretrained=True)

        self.encoder = nn.Sequential(*list(vgg16.features.children()))

        self.scales = []
        current_scale = 0
        for i, layer in enumerate(self.encoder):
            if isinstance(layer, torch.nn.MaxPool2d):
                current_scale += 1
            if i in self.hypercolumn_indices:
                self.scales.append(2 ** current_scale)

        ## adaptation layers, see off branches from fig.3 in S2DNet paper
        self.adaptation_layers = AdaptLayers2(vgg16_layers, **self.default_conf)

        # pose regression layers
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc_pose = nn.Linear(512, feat_dim)

    def forward(self, x, return_feature=False, isSingleStream=False, return_pose=True):
        """
        inference DFNet. It can regress camera pose as well as extract intermediate layer features.
            :param x: image blob (2B x C x H x W) two stream or (B x C x H x W) single stream
            :param return_feature: whether to return features as output
            :param isSingleStream: whether it's an single stream inference or siamese network inference
            :param upsampleH: feature upsample size H
            :param upsampleW: feature upsample size W
            :return feature_maps: (2, [B, C, H, W]) or (1, [B, C, H, W]) or None
            :return predict: [2B, 12] or [B, 12]
        """
        b, _, h, w = x.shape

        ### encoder ###
        feature_maps = []
        for i in range(len(self.encoder)):
            x = self.encoder[i](x)

            if i in self.hypercolumn_indices:
                feature = x.clone()
                feature_maps.append(feature)

                if i == self.hypercolumn_indices[-1]:
                    if not return_pose:
                        break

        ### extract and process intermediate features ###
        if return_feature:
            feature_maps, max_size = self.adaptation_layers(feature_maps)  # (3, [B, C, H', W']), H', W' are different in each layer

            if isSingleStream:  # not siamese network style inference
                feature_stacks = []
                for feature in feature_maps:
                    if feature.shape[-2:] != max_size:
                        feature = F.interpolate(feature, size=max_size, mode='bilinear', align_corners=False)
                    feature_stacks.append(feature)
                feature_maps = [torch.stack(feature_stacks)]  # (1, [3, B, C, H, W])
            else:  # siamese network style inference
                feature_stacks_t = []
                feature_stacks_r = []
                half = b // 2
                for feature in feature_maps:
                    if feature.shape[-2:] != max_size:
                        feature = F.interpolate(feature, size=max_size, mode='bilinear', align_corners=False)
                    feature_stacks_t.append(feature[:half])
                    feature_stacks_r.append(feature[half:])
                feature_stacks_t = torch.stack(feature_stacks_t)  # [B, C, H, W]
                feature_stacks_r = torch.stack(feature_stacks_r)  # [B, C, H, W]
                feature_maps = (feature_stacks_t, feature_stacks_r)  # (2, [B, C, H, W])
        else:
            feature_maps = None

        if not return_pose:
            return feature_maps, None

        ### pose regression head ###
        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        predict = self.fc_pose(x)

        return feature_maps, predict
