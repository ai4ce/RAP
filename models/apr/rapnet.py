from typing import List

import torch
from torch import nn
from torch.nn import functional as F

from utils.pose_utils import compute_rotation_matrix_from_ortho6d
from .backbone import build_backbone
from .transformer_encoder import Transformer

# efficientnet-B0 Layer Names and Channels
EB0_layers = {
    "reduction_1": 16,  # torch.Size([2, 16, 120, 213])
    "reduction_2": 24,  # torch.Size([2, 24, 60, 106])
    "reduction_3": 40,  # torch.Size([2, 40, 30, 53])
    "reduction_4": 112,  # torch.Size([2, 112, 15, 26])
    "reduction_5": 320,  # torch.Size([2, 320, 8, 13])
    "reduction_6": 1280,  # torch.Size([2, 1280, 8, 13])
}

EB3_layers = {
    "reduction_1": 24, # torch.Size([2, 24, 120, 213])
    "reduction_2": 32, # torch.Size([2, 32, 60, 106])
    "reduction_3": 48, # torch.Size([2, 48, 30, 53])
    "reduction_4": 136, # torch.Size([2, 136, 15, 26])
    "reduction_5": 384, # torch.Size([2, 384, 8, 13])
    "reduction_6": 1536, # torch.Size([2, 1536, 8, 13])
}


class PoseRegressor(nn.Module):
    """ A simple MLP to regress a pose component"""

    def __init__(self, decoder_dim, output_dim, use_prior=False):
        """
        decoder_dim: (int) the input dimension
        output_dim: (int) the outpur dimension
        use_prior: (bool) whether to use prior information
        """
        super().__init__()
        ch = 1024
        self.fc_h = nn.Linear(decoder_dim, ch)
        self.use_prior = use_prior
        if self.use_prior:
            self.fc_h_prior = nn.Linear(decoder_dim * 2, ch)
        self.fc_o = nn.Linear(ch, output_dim)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        """
        Forward pass
        """
        if self.use_prior:
            x = F.gelu(self.fc_h_prior(x))
        else:
            x = F.gelu(self.fc_h(x))

        return self.fc_o(x)


class AdaptLayers2(nn.Module):
    """Small adaptation layers.
    """

    def __init__(self, layer_sizes, hypercolumn_layers: List[str], output_dim: int = 128):
        """Initialize one adaptation layer for every extraction point.

        Args:
            hypercolumn_layers: The list of the hypercolumn layer names.
            output_dim: The output channel dimension.
        """
        super(AdaptLayers2, self).__init__()
        self.layers = []
        channel_sizes = [layer_sizes[name] for name in hypercolumn_layers]
        for i, l in enumerate(channel_sizes):
            layer = nn.Sequential(
                nn.Conv2d(l, 64, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, output_dim, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(output_dim),
            )
            self.layers.append(layer)
            self.add_module(f"adapt_layer_{i}", layer)  # ex: adapt_layer_0

    def forward(self, features: List[torch.tensor]):
        """Apply adaptation layers. # here is list of three levels of features
        """
        max_size = 0, 0
        for i, feature in enumerate(features):
            features[i] = getattr(self, f"adapt_layer_{i}")(feature)
            max_size = max(max_size[0], feature.shape[2]), max(max_size[1], feature.shape[3])
        return features, max_size


class RAPNet(nn.Module):
    """ RAPNet with EB0 backbone, feature levels can be customized """
    default_conf = {
        # 'hypercolumn_layers': ["reduction_1", "reduction_3", "reduction_6"],
        # 'hypercolumn_layers': ["reduction_1", "reduction_3", "reduction_5"],
        # 'hypercolumn_layers': ["reduction_2", "reduction_4", "reduction_6"],
        'hypercolumn_layers': ["reduction_4", "reduction_3"],
        # 'hypercolumn_layers': ["reduction_1"],
        'output_dim': 128,
    }

    def __init__(self, args=None):
        super().__init__()
        # Initialize architecture
        self.backbone = build_backbone(args)
        # self.feature_extractor = self.backbone_net.extract_endpoints
        # Position (t) and orientation (rot) encoders
        self.transformer_t = Transformer(args)
        self.transformer_rot = Transformer(args)
        decoder_dim = self.transformer_t.d_model
        self.pose_token_embed_t = nn.Parameter(torch.zeros((1, decoder_dim)), requires_grad=True)
        self.pose_token_embed_rot = nn.Parameter(torch.zeros((1, decoder_dim)), requires_grad=True)
        self.input_proj_t = nn.Conv2d(self.backbone.num_channels[0], decoder_dim, kernel_size=1)
        self.input_proj_rot = nn.Conv2d(self.backbone.num_channels[1], decoder_dim, kernel_size=1)
        # Regressors for position (t) and orientation (rot)
        self.regressor_head_t = PoseRegressor(decoder_dim, 3)
        self.regressor_head_rot = PoseRegressor(decoder_dim, 6, False)

        ## adaptation layers, see off branches from fig.3 in S2DNet paper
        self.adaptation_layers = AdaptLayers2(EB0_layers, **self.default_conf)

    def forward(self, x, return_feature=False, is_single_stream=False):
        """
        inference RAPNet. It can regress camera pose as well as extract intermediate layer features.
            :param x: image blob (2B x C x H x W) two stream or (B x C x H x W) single stream
            :param return_feature: whether to return features as output
            :param is_single_stream: whether it's a single stream inference or siamese network inference
            :return feature_maps: (2, [B, C, H, W]) or (1, [B, C, H, W]) or None
            :return predict: [2B, 12] or [B, 12]
        """
        B, C, H, W = x.shape
        feature_maps, (pos_t, pos_rot) = self.backbone(x)
        features_t, features_rot = feature_maps

        pose_token_embed_rot = self.pose_token_embed_rot.unsqueeze(1).expand(-1, B, -1)
        pose_token_embed_t = self.pose_token_embed_t.unsqueeze(1).expand(-1, B, -1)
        local_descs_t = self.transformer_t(self.input_proj_t(features_t), pos_t, pose_token_embed_t)  # torch.Size([20, 391, 256])
        local_descs_rot = self.transformer_rot(self.input_proj_rot(features_rot), pos_rot, pose_token_embed_rot)
        global_desc_t = local_descs_t[:, 0, :]
        global_desc_rot = local_descs_rot[:, 0, :]

        x_t = self.regressor_head_t(global_desc_t)  # torch.Size([8, 3])
        x_rot = self.regressor_head_rot(global_desc_rot)  # torch.Size([8, 6])
        x_rot = compute_rotation_matrix_from_ortho6d(x_rot)
        x_t = x_t.reshape(B, 3, 1)
        x_rot = x_rot.reshape(B, 3, 3)
        pose = torch.cat((x_rot, x_t), dim=2)
        predicted_pose = pose.reshape(B, 12)  # torch.Size([8, 12])

        ### extract and process intermediate features ###
        if return_feature:
            feature_maps, max_size = self.adaptation_layers(feature_maps)  # (3, [B, C, H', W']), H', W' are different in each layer

            if is_single_stream:  # not siamese network style inference
                feature_stacks = []
                for feature in feature_maps:
                    if feature.shape[-2:] != max_size:
                        feature = F.interpolate(feature, size=max_size, mode='bilinear', align_corners=False)
                    feature_stacks.append(feature)
                feature_maps = [torch.stack(feature_stacks)]  # (1, [3, B, C, H, W])
            else:  # siamese network style inference
                feature_stacks_t = []
                feature_stacks_r = []
                half = B // 2
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

        return feature_maps, predicted_pose


def main():
    """
    test model
    """
    from torchsummary import summary
    feat_model = RAPNet()
    summary(feat_model, (3, 240, 427))


if __name__ == '__main__':
    main()
