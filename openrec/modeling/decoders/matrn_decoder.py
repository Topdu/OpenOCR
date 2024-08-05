"""This code is refer from:
https://github.com/byeonghu-na/MATRN
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from openrec.modeling.decoders.abinet_decoder import BCNLanguage, PositionAttention, _get_length
from openrec.modeling.decoders.nrtr_decoder import PositionalEncoding, TransformerBlock


class BaseSemanticVisual_backbone_feature(nn.Module):

    def __init__(self,
                 d_model=512,
                 nhead=8,
                 num_layers=4,
                 dim_feedforward=2048,
                 dropout=0.0,
                 alignment_mask_example_prob=0.9,
                 alignment_mask_candidate_prob=0.9,
                 alignment_num_vis_mask=10,
                 max_length=25,
                 num_classes=37):
        super().__init__()
        self.mask_example_prob = alignment_mask_example_prob
        self.mask_candidate_prob = alignment_mask_candidate_prob  #ifnone(config.model_alignment_mask_candidate_prob, 0.9)
        self.num_vis_mask = alignment_num_vis_mask
        self.nhead = nhead

        self.d_model = d_model
        self.max_length = max_length + 1  # additional stop token

        self.model1 = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                attention_dropout_rate=dropout,
                residual_dropout_rate=dropout,
                with_self_attn=True,
                with_cross_attn=False,
            ) for i in range(num_layers)
        ])
        self.pos_encoder_tfm = PositionalEncoding(dim=d_model,
                                                  dropout=0,
                                                  max_len=1024)

        self.model2_vis = PositionAttention(
            max_length=self.max_length,  # additional stop token
            in_channels=d_model,
            num_channels=d_model // 8,
            mode='nearest',
        )
        self.cls_vis = nn.Linear(d_model, num_classes)
        self.cls_sem = nn.Linear(d_model, num_classes)
        self.w_att = nn.Linear(2 * d_model, d_model)

        v_token = torch.empty((1, d_model))
        self.v_token = nn.Parameter(v_token)
        torch.nn.init.uniform_(self.v_token, -0.001, 0.001)

        self.cls = nn.Linear(d_model, num_classes)

    def forward(self, l_feature, v_feature, lengths_l=None, v_attn=None):
        """
        Args:
            l_feature: (N, T, E) where T is length, N is batch size and d is dim of model
            v_feature: (N, E, H, W)
            lengths_l: (N,)
            v_attn: (N, T, H, W)
            l_logits: (N, T, C)
            texts: (N, T, C)
        """

        N, E, H, W = v_feature.size()
        v_feature = v_feature.flatten(2, 3).transpose(1, 2)  #(N, H*W, E)
        v_attn = v_attn.flatten(2, 3)  # (N, T, H*W)
        if self.training:
            for idx, length in enumerate(lengths_l):
                if np.random.random() <= self.mask_example_prob:
                    l_idx = np.random.randint(int(length))
                    v_random_idx = v_attn[idx, l_idx].argsort(
                        descending=True).cpu().numpy()[:self.num_vis_mask, ]
                    v_random_idx = v_random_idx[np.random.random(
                        v_random_idx.shape) <= self.mask_candidate_prob]
                    v_feature[idx, v_random_idx] = self.v_token

        zeros = v_feature.new_zeros((N, H * W, E))  # (N, H*W, E)
        base_pos = self.pos_encoder_tfm(zeros)  # (N, H*W, E)
        base_pos = torch.bmm(v_attn, base_pos)  # (N, T, E)

        l_feature = l_feature + base_pos

        sv_feature = torch.cat((v_feature, l_feature), dim=1)  # (H*W+T, N, E)
        for decoder_layer in self.model1:
            sv_feature = decoder_layer(sv_feature)  # (H*W+T, N, E)

        sv_to_v_feature = sv_feature[:, :H * W]  # (N, H*W, E)
        sv_to_s_feature = sv_feature[:, H * W:]  # (N, T, E)

        sv_to_v_feature = sv_to_v_feature.transpose(1, 2).reshape(N, E, H, W)
        sv_to_v_feature, _ = self.model2_vis(sv_to_v_feature)  # (N, T, E)
        sv_to_v_logits = self.cls_vis(sv_to_v_feature)  # (N, T, C)
        pt_v_lengths = _get_length(sv_to_v_logits)  # (N,)

        sv_to_s_logits = self.cls_sem(sv_to_s_feature)  # (N, T, C)
        pt_s_lengths = _get_length(sv_to_s_logits)  # (N,)

        f = torch.cat((sv_to_v_feature, sv_to_s_feature), dim=2)
        f_att = torch.sigmoid(self.w_att(f))
        output = f_att * sv_to_v_feature + (1 - f_att) * sv_to_s_feature

        logits = self.cls(output)  # (N, T, C)
        pt_lengths = _get_length(logits)

        return {
            'logits': logits,
            'pt_lengths': pt_lengths,
            'v_logits': sv_to_v_logits,
            'pt_v_lengths': pt_v_lengths,
            's_logits': sv_to_s_logits,
            'pt_s_lengths': pt_s_lengths,
            'name': 'alignment'
        }


class MATRNDecoder(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 nhead=8,
                 num_layers=3,
                 dim_feedforward=2048,
                 dropout=0.1,
                 max_length=25,
                 iter_size=3,
                 **kwargs):
        super().__init__()
        self.max_length = max_length + 1
        d_model = in_channels
        self.pos_encoder = PositionalEncoding(dropout=0.1, dim=d_model)
        self.encoder = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                attention_dropout_rate=dropout,
                residual_dropout_rate=dropout,
                with_self_attn=True,
                with_cross_attn=False,
            ) for _ in range(num_layers)
        ])
        self.decoder = PositionAttention(
            max_length=self.max_length,  # additional stop token
            in_channels=d_model,
            num_channels=d_model // 8,
            mode='nearest',
        )
        self.out_channels = out_channels
        self.cls = nn.Linear(d_model, self.out_channels)
        self.iter_size = iter_size
        if iter_size > 0:
            self.language = BCNLanguage(
                d_model=d_model,
                nhead=nhead,
                num_layers=4,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                max_length=max_length,
                num_classes=self.out_channels,
            )
            # alignment
            self.semantic_visual = BaseSemanticVisual_backbone_feature(
                d_model=d_model,
                nhead=nhead,
                num_layers=2,
                dim_feedforward=dim_feedforward,
                max_length=max_length,
                num_classes=self.out_channels)

    def forward(self, x, data=None):
        # bs, c, h, w
        x = x.permute([0, 2, 3, 1])  # bs, h, w, c
        _, H, W, C = x.shape
        # assert H % 8 == 0 and W % 16 == 0, 'The height and width should be multiples of 8 and 16.'
        feature = x.flatten(1, 2)  # bs, h*w, c
        feature = self.pos_encoder(feature)  # bs, h*w, c
        for encoder_layer in self.encoder:
            feature = encoder_layer(feature)
        # bs, h*w, c
        feature = feature.reshape([-1, H, W, C]).permute(0, 3, 1,
                                                         2)  # bs, c, h, w
        v_feature, v_attn_input = self.decoder(feature)  # (bs[N], T, E)
        vis_logits = self.cls(v_feature)  # (bs[N], T, E)
        align_lengths = _get_length(vis_logits)
        align_logits = vis_logits
        all_l_res, all_a_res = [], []
        for _ in range(self.iter_size):
            tokens = F.softmax(align_logits, dim=-1)
            lengths = torch.clamp(
                align_lengths, 2,
                self.max_length)  # TODO: move to language model
            l_feature, l_logits = self.language(tokens, lengths)
            all_l_res.append(l_logits)
            # alignment
            lengths_l = _get_length(l_logits)
            lengths_l.clamp_(2, self.max_length)

            a_res = self.semantic_visual(l_feature,
                                         feature,
                                         lengths_l=lengths_l,
                                         v_attn=v_attn_input)

            a_v_res = a_res['v_logits']
            # {'logits': a_res['v_logits'], 'pt_lengths': a_res['pt_v_lengths'], 'loss_weight': a_res['loss_weight'],
            #               'name': 'alignment'}
            all_a_res.append(a_v_res)
            a_s_res = a_res['s_logits']
            # {'logits': a_res['s_logits'], 'pt_lengths': a_res['pt_s_lengths'], 'loss_weight': a_res['loss_weight'],
            #               'name': 'alignment'}
            align_logits = a_res['logits']
            all_a_res.append(a_s_res)
            all_a_res.append(align_logits)
            align_lengths = a_res['pt_lengths']
        if self.training:
            return {
                'align': all_a_res,
                'lang': all_l_res,
                'vision': vis_logits
            }
        else:
            return F.softmax(align_logits, -1)
