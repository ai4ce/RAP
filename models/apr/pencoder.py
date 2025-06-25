"""
Code for the position encoding of TransPoseNet
 code is based on https://github.com/facebookresearch/detr/tree/master/models with the following modifications:
- changed to learn also the position of a learned pose token
"""

import torch
from torch import nn


class PositionEmbeddingLearnedWithPoseToken(nn.Module):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, num_pos_feats=256, device="cuda"):  # 128
        super().__init__()
        self.row_embed = nn.Embedding(256, num_pos_feats)
        self.col_embed = nn.Embedding(256, num_pos_feats)
        self.pose_token_embed = nn.Embedding(256, num_pos_feats)
        self.reset_parameters()
        self.p = torch.tensor(0, device=device)

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)
        nn.init.uniform_(self.pose_token_embed.weight)

    def forward(self, x):
        b, c, h, w = x.shape
        i = torch.arange(1, w + 1, device=x.device)
        j = torch.arange(1, h + 1, device=x.device)
        p_emb = self.pose_token_embed(self.p)[None, None, ...].expand(b, 2, -1).reshape(b, -1)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        # embed of position in the activation map
        m_emb = torch.cat((
            x_emb.unsqueeze(0).expand(h, -1, -1),
            y_emb.unsqueeze(1).expand(-1, w, -1),
        ), dim=-1).permute(2, 0, 1).unsqueeze(0).expand(b, -1, -1, -1)
        return p_emb, m_emb


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, num_pos_feats=256, device="cuda"):
        super().__init__()
        self.row_embed = nn.Embedding(256, num_pos_feats)
        self.col_embed = nn.Embedding(256, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, x):
        b, _, h, w = x.shape
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).expand(h, -1, -1),
            y_emb.unsqueeze(1).expand(-1, w, -1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).expand(b, -1, -1, -1)
        return pos


def build_position_encoding(config):
    N_steps = config.hidden_dim // 2
    if config.learn_embedding_with_pose_token:
        position_embedding = PositionEmbeddingLearnedWithPoseToken(N_steps, config.device)
    else:
        position_embedding = PositionEmbeddingLearned(N_steps, config.device)
    return position_embedding
