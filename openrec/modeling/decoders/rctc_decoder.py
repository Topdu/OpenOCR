import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_

from openrec.modeling.common import Block


class RCTCDecoder(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels=6625,
                 return_feats=False,
                 **kwargs):
        super(RCTCDecoder, self).__init__()
        self.char_token = nn.Parameter(
            torch.zeros([1, 1, in_channels], dtype=torch.float32),
            requires_grad=True,
        )
        trunc_normal_(self.char_token, mean=0, std=0.02)
        self.fc = nn.Linear(
            in_channels,
            out_channels,
            bias=True,
        )
        self.fc_kv = nn.Linear(
            in_channels,
            2 * in_channels,
            bias=True,
        )
        self.w_atten_block = Block(dim=in_channels,
                                   num_heads=in_channels // 32,
                                   mlp_ratio=4.0,
                                   qkv_bias=False)
        self.out_channels = out_channels
        self.return_feats = return_feats

    def forward(self, x, data=None):

        B, C, H, W = x.shape
        x = self.w_atten_block(x.permute(0, 2, 3,
                                         1).reshape(-1, W, C)).reshape(
                                             B, H, W, C).permute(0, 3, 1, 2)
        # B, D, 8, 32
        x_kv = self.fc_kv(x.flatten(2).transpose(1, 2)).reshape(
            B, H * W, 2, C).permute(2, 0, 3, 1)  # 2, b, c, hw
        x_k, x_v = x_kv.unbind(0)  # b, c, hw
        char_token = self.char_token.tile([B, 1, 1])
        attn_ctc2d = char_token @ x_k  # b, 1, hw
        attn_ctc2d = attn_ctc2d.reshape([-1, 1, H, W])
        attn_ctc2d = F.softmax(attn_ctc2d, 2)  # b, 1, h, w
        attn_ctc2d = attn_ctc2d.permute(0, 3, 1, 2)  # b, w, 1, h
        x_v = x_v.reshape(B, C, H, W)
        # B, W, H, C
        feats = attn_ctc2d @ x_v.permute(0, 3, 2, 1)  # b, w, 1, c
        feats = feats.squeeze(2)  # b, w, c

        predicts = self.fc(feats)

        if self.return_feats:
            result = (feats, predicts)
        else:
            result = predicts

        if not self.training:
            predicts = F.softmax(predicts, dim=2)
            result = predicts

        return result
