"""
Code for the backbone of TransPoseNet
Backbone code is based on https://github.com/facebookresearch/detr/tree/master/models with the following modifications:
- use efficient-net as backbone and extract different activation maps from different reduction maps
- change learned encoding to have a learned token for the pose
"""

from efficientnet_pytorch import EfficientNet
from torch import nn

from .pencoder import build_position_encoding


class BackboneBase(nn.Module):
    def __init__(self, backbone: nn.Module, reduction, reduction_map):
        super().__init__()
        self.body = backbone
        self.reductions = reduction
        # self.reduction_map = {"reduction_3": 40, "reduction_4": 112, "reduction_6":1280}
        # self.reduction_map = {"reduction_3": 40, "reduction_4": 112, "reduction_5":320}
        self.num_channels = [reduction_map[reduction] for reduction in self.reductions]

    def forward(self, xs):
        xs = self.body.extract_endpoints(xs)
        out = []
        for name in self.reductions:
            out.append(xs[name])
        return out


class Backbone(BackboneBase):
    def __init__(self, reduction):
        backbone = EfficientNet.from_pretrained("efficientnet-b0")
        reduction_map = {"reduction_3": 40, "reduction_4": 112}  # dict is ordered after Python 3.6
        super().__init__(backbone, reduction, reduction_map)

class BackboneE3(BackboneBase):
    def __init__(self, reduction):
        backbone = EfficientNet.from_pretrained("efficientnet-b3")
        reduction_map = {"reduction_1": 24, "reduction_4": 136, "reduction_5": 384}
        super().__init__(backbone, reduction, reduction_map)


class Joiner(nn.Sequential):
    def forward(self, xs):
        xs = self[0](xs)
        pos = []
        for x in xs:
            # position encoding
            ret = self[1](x)
            pos.append(ret)
        return xs, pos


def build_backbone(config):
    position_embedding = build_position_encoding(config)
    backbone = Backbone(config.reduction)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model


def build_backbone_E3(config):
    position_embedding = build_position_encoding(config)
    backbone = BackboneE3(config.reduction)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model
