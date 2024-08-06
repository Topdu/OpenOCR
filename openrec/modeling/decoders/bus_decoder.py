"""This code is refer from:
https://github.com/jjwei66/BUSNet
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .nrtr_decoder import PositionalEncoding, TransformerBlock
from .abinet_decoder import _get_mask, _get_length


class BUSDecoder(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 nhead=8,
                 num_layers=4,
                 dim_feedforward=2048,
                 dropout=0.1,
                 max_length=25,
                 ignore_index=100,
                 pretraining=False,
                 detach=True):
        super().__init__()
        d_model = in_channels
        self.ignore_index = ignore_index
        self.pretraining = pretraining
        self.d_model = d_model
        self.detach = detach
        self.max_length = max_length + 1  # additional stop token
        self.out_channels = out_channels
        # --------------------------------------------------------------------------
        # decoder specifics
        self.proj = nn.Linear(out_channels, d_model, False)
        self.token_encoder = PositionalEncoding(dropout=0.1,
                                                dim=d_model,
                                                max_len=self.max_length)
        self.pos_encoder = PositionalEncoding(dropout=0.1,
                                              dim=d_model,
                                              max_len=self.max_length)

        self.decoder = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                attention_dropout_rate=dropout,
                residual_dropout_rate=dropout,
                with_self_attn=False,
                with_cross_attn=True,
            ) for i in range(num_layers)
        ])

        v_mask = torch.empty((1, 1, d_model))
        l_mask = torch.empty((1, 1, d_model))
        self.v_mask = nn.Parameter(v_mask)
        self.l_mask = nn.Parameter(l_mask)
        torch.nn.init.uniform_(self.v_mask, -0.001, 0.001)
        torch.nn.init.uniform_(self.l_mask, -0.001, 0.001)

        v_embeding = torch.empty((1, 1, d_model))
        l_embeding = torch.empty((1, 1, d_model))
        self.v_embeding = nn.Parameter(v_embeding)
        self.l_embeding = nn.Parameter(l_embeding)
        torch.nn.init.uniform_(self.v_embeding, -0.001, 0.001)
        torch.nn.init.uniform_(self.l_embeding, -0.001, 0.001)
        self.cls = nn.Linear(d_model, out_channels)

    def forward_decoder(self, q, x, mask=None):
        for decoder_layer in self.decoder:
            q = decoder_layer(q, x, cross_mask=mask)
        output = q  # (N, T, E)
        logits = self.cls(output)  # (N, T, C)
        return logits

    def forward(self, img_feat, data=None):
        """
        Args:
            tokens: (N, T, C) where T is length, N is batch size and C is classes number
            lengths: (N,)
        """
        img_feat = img_feat + self.v_embeding
        B, L, C = img_feat.shape

        # --------------------------------------------------------------------------
        # decoder procedure
        T = self.max_length
        zeros = img_feat.new_zeros((B, T, C))
        zeros_len = img_feat.new_zeros(B)
        query = self.pos_encoder(zeros)

        # 1. vision decode
        v_embed = torch.cat((img_feat, self.l_mask.repeat(B, T, 1)),
                            dim=1)  # v
        padding_mask = _get_mask(
            self.max_length + zeros_len,
            self.max_length)  # 对tokens长度以外的padding # B, maxlen maxlen
        v_mask = torch.zeros((1, 1, self.max_length, L),
                             device=img_feat.device).tile([B, 1, 1,
                                                           1])  # maxlen L
        mask = torch.cat((v_mask, padding_mask), 3)
        v_logits = self.forward_decoder(query, v_embed, mask=mask)

        # 2. language decode
        if self.training and self.pretraining:
            tgt = torch.where(data[0] == self.ignore_index, 0, data[0])
            tokens = F.one_hot(tgt, num_classes=self.out_channels)
            tokens = tokens.float()
            lengths = data[-1]
        else:
            tokens = torch.softmax(v_logits, dim=-1)
            lengths = _get_length(v_logits)
            tokens = tokens.detach()
        token_embed = self.proj(tokens)  # (N, T, E)
        token_embed = self.token_encoder(token_embed)  # (T, N, E)
        token_embed = token_embed + self.l_embeding

        padding_mask = _get_mask(lengths,
                                 self.max_length)  # 对tokens长度以外的padding
        mask = torch.cat((v_mask, padding_mask), 3)
        l_embed = torch.cat((self.v_mask.repeat(B, L, 1), token_embed), dim=1)
        l_logits = self.forward_decoder(query, l_embed, mask=mask)

        # 3. vision language decode
        vl_embed = torch.cat((img_feat, token_embed), dim=1)
        vl_logits = self.forward_decoder(query, vl_embed, mask=mask)

        if self.training:
            return {'align': [vl_logits], 'lang': l_logits, 'vision': v_logits}
        else:
            return F.softmax(vl_logits, -1)
