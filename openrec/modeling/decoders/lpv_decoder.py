import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from .abinet_decoder import PositionAttention
from .nrtr_decoder import PositionalEncoding, TransformerBlock


class Trans(nn.Module):

    def __init__(self, dim, nhead, dim_feedforward, dropout, num_layers):
        super().__init__()
        self.d_model = dim
        self.nhead = nhead

        self.pos_encoder = PositionalEncoding(dropout=0.0,
                                              dim=self.d_model,
                                              max_len=512)

        self.transformer = nn.ModuleList([
            TransformerBlock(
                dim,
                nhead,
                dim_feedforward,
                attention_dropout_rate=dropout,
                residual_dropout_rate=dropout,
                with_self_attn=True,
                with_cross_attn=False,
            ) for i in range(num_layers)
        ])

    def forward(self, feature, attn_map=None, use_mask=False):
        n, c, h, w = feature.shape
        feature = feature.flatten(2).transpose(1, 2)

        if use_mask:
            _, t, h, w = attn_map.shape
            location_mask = (attn_map.view(n, t, -1).transpose(1, 2) >
                             0.05).type(torch.float)  # n,hw,t
            location_mask = location_mask.bmm(location_mask.transpose(
                1, 2))  # n, hw, hw
            location_mask = location_mask.new_zeros(
                (h * w, h * w)).masked_fill(location_mask > 0, float('-inf'))
            location_mask = location_mask.unsqueeze(1)  # n, 1, hw, hw
        else:
            location_mask = None

        feature = self.pos_encoder(feature)
        for layer in self.transformer:
            feature = layer(feature, self_mask=location_mask)
        feature = feature.transpose(1, 2).view(n, c, h, w)
        return feature, location_mask


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class LPVDecoder(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_layer=3,
                 max_len=25,
                 use_mask=False,
                 dim_feedforward=1024,
                 nhead=8,
                 dropout=0.1,
                 trans_layer=2):
        super().__init__()
        self.use_mask = use_mask
        self.max_len = max_len
        attn_layer = PositionAttention(max_length=max_len + 1,
                                       mode='nearest',
                                       in_channels=in_channels,
                                       num_channels=in_channels // 8)
        trans_layer = Trans(dim=in_channels,
                            nhead=nhead,
                            dim_feedforward=dim_feedforward,
                            dropout=dropout,
                            num_layers=trans_layer)
        cls_layer = nn.Linear(in_channels, out_channels - 2)

        self.attention = _get_clones(attn_layer, num_layer)
        self.trans = _get_clones(trans_layer, num_layer - 1)
        self.cls = _get_clones(cls_layer, num_layer)

    def forward(self, x, data=None):
        if data is not None:
            max_len = data[1].max()
        else:
            max_len = self.max_len
        features = x  # (N, E, H, W)

        attn_vecs, attn_scores_map = self.attention[0](features)
        attn_vecs = attn_vecs[:, :max_len + 1, :]
        if not self.training:
            for i in range(1, len(self.attention)):
                features, mask = self.trans[i - 1](features,
                                                   attn_scores_map,
                                                   use_mask=self.use_mask)
                attn_vecs, attn_scores_map = self.attention[i](
                    features, attn_vecs)  # (N, T, E), (N, T, H, W)
            return F.softmax(self.cls[-1](attn_vecs), -1)
        else:
            logits = []
            logit = self.cls[0](attn_vecs)  # (N, T, C)
            logits.append(logit)
            for i in range(1, len(self.attention)):
                features, mask = self.trans[i - 1](features,
                                                   attn_scores_map,
                                                   use_mask=self.use_mask)
                attn_vecs, attn_scores_map = self.attention[i](
                    features, attn_vecs)  # (N, T, E), (N, T, H, W)
                logit = self.cls[i](attn_vecs)  # (N, T, C)
                logits.append(logit)
            return logits
