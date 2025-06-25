from torch import nn

from models.gs.basic_mlp import LinModule
from models.gs.embedder import *


class ColorNet(nn.Module):
    def __init__(self,
                 fin_dim,
                 pin_dim,
                 view_dim,
                 pfin_dim,
                 en_dims,
                 de_dims,
                 multires,
                 pre_compc=False,
                 cde_dims=None,
                 use_pencoding=(False, False),  # position viewdir
                 weight_norm=False,
                 weight_xavier=True,
                 use_drop_out=False,
                 use_decode_with_pos=False,
                 ):
        super().__init__()
        self.pre_compc = pre_compc
        self.use_pencoding = use_pencoding
        self.embed_fns = []
        self.all_features_decoded_cache = None
        self.use_decode_with_pos = use_decode_with_pos
        if use_pencoding[0]:
            embed_fn, input_ch = get_embedder(multires[0])
            pin_dim = input_ch
            self.embed_fns.append(embed_fn)
        else:
            self.embed_fns.append(None)

        if use_pencoding[1]:
            embed_fn, input_ch = get_embedder(multires[1])
            view_dim = input_ch
            self.embed_fns.append(embed_fn)
        else:
            self.embed_fns.append(None)

        self.encoder = LinModule(fin_dim + pin_dim + pfin_dim, fin_dim, en_dims, multires[0],
                                 act_fun=nn.ReLU(inplace=True), weight_norm=weight_norm, weight_xavier=weight_xavier)
        self.decoder = LinModule(fin_dim * 2, fin_dim, de_dims, multires[0],
                                 act_fun=nn.ReLU(inplace=True), weight_norm=weight_norm, weight_xavier=weight_xavier)
        if self.pre_compc:
            # view_dim=3
            self.color_decoder = LinModule(fin_dim + view_dim, 3, cde_dims, multires[0],
                                           act_fun=nn.ReLU(inplace=True), weight_norm=weight_norm, weight_xavier=weight_xavier)
            # self.color_decoder=lin_module(fin_dim+pin_dim,3,cde_dims,multires[0],act_fun=nn.ReLU(),weight_norm=weight_norm,weight_xavier=weight_xavier)
        self.use_drop_out = use_drop_out

        if use_drop_out:
            self.drop_outs = [nn.Dropout(0.1, inplace=True)]

    def forward(self, points_xyz, intrinsic_features, appearance_features,
                view_direction=None, inter_weight=1.0, store_cache=False):
        points_xyz_orig = points_xyz
        if self.use_drop_out:
            appearance_features = self.drop_outs[0](appearance_features)
        if self.use_pencoding[0]:
            if self.use_decode_with_pos:
                points_xyz_orig = points_xyz.clone()
            points_xyz = self.embed_fns[0](points_xyz)

        if self.use_pencoding[1]:
            view_direction = self.embed_fns[1](view_direction)
            # view_direction=self.embed_fn(view_direction)
        p_num = intrinsic_features.shape[0]
        intrinsic_features = intrinsic_features.reshape([p_num, -1])

        appearance_features *= inter_weight
        all_features = torch.cat([points_xyz, appearance_features, intrinsic_features], dim=1)
        # inx=torch.cat([inp,inf],dim=1)
        all_features_encoded = self.encoder(all_features)
        all_features_decoded = self.decoder(torch.cat([all_features_encoded, intrinsic_features], dim=1))
        if store_cache:
            self.all_features_decoded_cache = all_features_decoded
        else:
            self.all_features_decoded_cache = None

        if self.pre_compc:
            if self.use_decode_with_pos:
                outc = self.color_decoder(torch.cat([all_features_decoded, points_xyz_orig], dim=1))
            else:
                outc = self.color_decoder(torch.cat([all_features_decoded, view_direction], dim=1))  # view_direction
            return outc
        return all_features_decoded.reshape([p_num, -1, 3])

    def forward_cache(self, points_xyz, view_direction=None):
        points_xyz_orig = points_xyz
        if self.use_pencoding[0]:
            if self.use_decode_with_pos:
                points_xyz_orig = points_xyz.clone()
            points_xyz = self.embed_fns[0](points_xyz)
        if self.use_pencoding[1]:
            view_direction = self.embed_fns[1](view_direction)
        p_num = points_xyz.shape[0]
        if self.pre_compc:
            if self.use_decode_with_pos:
                outc = self.color_decoder(torch.cat([self.all_features_decoded_cache, points_xyz_orig], dim=1))
            else:
                outc = self.color_decoder(torch.cat([self.all_features_decoded_cache, view_direction], dim=1))  # view_direction
            return outc
        return self.all_features_decoded_cache.reshape([p_num, -1, 3])
