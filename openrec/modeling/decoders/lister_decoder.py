"""This code is refer from:
https://github.com/AlibabaResearch/AdvancedLiterateMachinery/blob/main/OCR/LISTER
"""

# Copyright (2023) Alibaba Group and its affiliates
# --------------------------------------------------------
# To decode arbitrary-length text images.
# --------------------------------------------------------
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_

from openrec.modeling.encoders.focalsvtr import FocalNetBlock


class LocalSelfAttention(nn.Module):

    def __init__(self,
                 feat_dim,
                 nhead,
                 window_size: int,
                 add_pos_bias=False,
                 qkv_drop=0.0,
                 proj_drop=0.0,
                 mlm=False):
        super().__init__()
        assert feat_dim % nhead == 0
        self.q_fc = nn.Linear(feat_dim, feat_dim)
        self.kv_fc = nn.Linear(feat_dim, feat_dim * 2)

        self.nhead = nhead
        self.head_dim = feat_dim // nhead
        self.window_size = window_size
        if add_pos_bias:
            self.kv_pos_bias = nn.Parameter(torch.zeros(window_size, feat_dim))
            trunc_normal_(self.kv_pos_bias, std=.02)
        else:
            self.kv_pos_bias = None
        self.qkv_dropout = nn.Dropout(qkv_drop)

        self.proj = nn.Linear(feat_dim, feat_dim)
        self.proj_dropout = nn.Dropout(proj_drop)
        self.mlm = mlm
        if mlm:
            print('Use mlm.')

    def _gen_t_index(self, real_len, device):
        idx = torch.stack([
            torch.arange(real_len, dtype=torch.long, device=device) + st
            for st in range(self.window_size)
        ]).t()  # [T, w]
        return idx

    def _apply_attn_mask(self, attn_score):
        attn_score[:, :, :, :, self.window_size // 2] = float('-inf')
        return attn_score

    def forward(self, x, mask):
        """
        Args:
            x: [b, T, C]
            mask: [b, T]
        """
        b, T, C = x.size()
        # mask with 0
        x = x * mask.unsqueeze(-1)

        q = self.q_fc(self.qkv_dropout(x))  # [b, T, C]
        pad_l = pad_r = self.window_size // 2
        x_pad = F.pad(x, (0, 0, pad_l, pad_r))  # [b, T+w, C]
        # organize the window-based kv
        b_idx = torch.arange(b, dtype=torch.long,
                             device=x.device).contiguous().view(b, 1, 1)
        t_idx = self._gen_t_index(T, x.device).unsqueeze(0)
        x_pad = x_pad[b_idx, t_idx]  # [b, T, w, C]
        if self.kv_pos_bias is not None:
            x_pad = self.qkv_dropout(
                x_pad + self.kv_pos_bias.unsqueeze(0).unsqueeze(1))
        else:
            x_pad = self.qkv_dropout(x_pad)
        kv = self.kv_fc(x_pad)  # [b, T, w, 2*C]
        k, v = kv.chunk(2, -1)  # both are [b, T, w, C]
        # multi-head splitting
        q = q.contiguous().view(b, T, self.nhead, -1)  # [b, T, h, C/h]
        k = k.contiguous().view(b, T, self.window_size, self.nhead,
                                -1).transpose(2, 3)  # [b, T, h, w, C/h]
        v = v.contiguous().view(b, T, self.window_size, self.nhead,
                                -1).transpose(2, 3)
        # calculate attention scores
        # the scaling of qk refers to: https://kexue.fm/archives/8823
        alpha = q.unsqueeze(3).matmul(
            k.transpose(-1, -2) / self.head_dim *
            math.log(self.window_size))  # [b, T, h, 1, w]
        if self.mlm:
            alpha = self._apply_attn_mask(alpha)
        alpha = alpha.softmax(-1)
        output = alpha.matmul(v).squeeze(-2).contiguous().view(b, T,
                                                               -1)  # [b, T, C]
        output = self.proj_dropout(self.proj(output))
        output = output * mask.unsqueeze(-1)
        return output


class LocalAttentionBlock(nn.Module):

    def __init__(self,
                 feat_dim,
                 nhead,
                 window_size,
                 add_pos_bias: bool,
                 drop=0.0,
                 proj_drop=0.0,
                 init_values=1e-6,
                 mlm=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(feat_dim)
        self.sa = LocalSelfAttention(feat_dim,
                                     nhead,
                                     window_size,
                                     add_pos_bias,
                                     drop,
                                     proj_drop,
                                     mlm=mlm)
        self.norm2 = nn.LayerNorm(feat_dim)
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, feat_dim * 4),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(feat_dim * 4, feat_dim),
            nn.Dropout(drop),
        )
        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones(feat_dim),
                                        requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones(feat_dim),
                                        requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = 1.0, 1.0

    def forward(self, x, mask):
        x = x + self.gamma_1 * self.sa(self.norm1(x), mask)
        x = x + self.gamma_2 * self.mlp(self.norm2(x))
        x = x * mask.unsqueeze(-1)
        return x


class LocalAttentionModule(nn.Module):

    def __init__(self,
                 feat_dim,
                 nhead,
                 window_size,
                 num_layers,
                 drop_rate=0.0,
                 proj_drop_rate=0.0,
                 detach_grad=False,
                 mlm=False):
        super().__init__()
        self.attn_blocks = nn.ModuleList([
            LocalAttentionBlock(
                feat_dim,
                nhead,
                window_size,
                add_pos_bias=(i == 0),
                drop=drop_rate,
                proj_drop=proj_drop_rate,
                mlm=mlm,
            ) for i in range(num_layers)
        ])

        self.detach_grad = detach_grad

    def forward(self, x, mask):
        if self.detach_grad:
            x = x.detach()
        for blk in self.attn_blocks:
            x = blk(x, mask)
        return x


def softmax_m1(x: torch.Tensor, dim: int):
    # for x >= 0
    fx = x.exp() - 1
    fx = fx / fx.sum(dim, keepdim=True)
    return fx


class BilinearLayer(nn.Module):

    def __init__(self, in1, in2, out, bias=True):
        super(BilinearLayer, self).__init__()
        self.weight = nn.Parameter(torch.randn(out, in1, in2))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out))
        else:
            self.bias = None
        torch.nn.init.xavier_normal_(self.weight, 0.1)

    def forward(self, x1, x2):
        '''
        input:
        x1: [b, T1, in1]
        x2: [b, T2, in2]
        output:
        y: [b, T1, T2, out]
        '''
        y = torch.einsum('bim,omn->bino', x1, self.weight)  # [b, T1, in2, out]
        y = torch.einsum('bino,bjn->bijo', y, x2)  # [b, T1, T2, out]
        if self.bias is not None:
            y = y + self.bias.contiguous().view(1, 1, 1, -1)
        return y


class NeighborDecoder(nn.Module):
    """Find neighbors for each character In this version, each iteration shares
    the same decoder with the local vision decoder."""

    def __init__(self,
                 num_classes,
                 feat_dim,
                 max_len=1000,
                 detach_grad=False,
                 **kwargs):
        super().__init__()
        self.eos_emb = nn.Parameter(torch.ones(feat_dim))
        trunc_normal_(self.eos_emb, std=.02)
        self.q_fc = nn.Linear(feat_dim, feat_dim, bias=True)
        self.k_fc = nn.Linear(feat_dim, feat_dim)

        self.neighbor_navigator = BilinearLayer(feat_dim, feat_dim, 1)

        self.vis_cls = nn.Linear(feat_dim, num_classes)

        self.p_threshold = 0.6
        self.max_len = max_len or 1000  # to avoid endless loop
        self.max_ch = max_len or 1000

        self.detach_grad = detach_grad
        self.attn_scaling = kwargs['attn_scaling']

    def align_chars(self, start_map, nb_map, max_ch=None):
        if self.training:
            assert max_ch is not None
        max_ch = max_ch or self.max_ch  # required during training to be efficient
        b, N = nb_map.shape[:2]

        char_map = start_map  # [b, N]
        all_finished = torch.zeros(b, dtype=torch.long, device=nb_map.device)
        char_maps = []
        char_masks = []
        for i in range(max_ch):
            char_maps.append(char_map)
            char_mask = (all_finished == 0).float()
            char_masks.append(char_mask)
            if i == max_ch - 1:
                break
            all_finished = all_finished + (char_map[:, -1] >
                                           self.p_threshold).long()
            if not self.training:
                # check if end
                if (all_finished > 0).sum().item() == b:
                    break
            if self.training:
                char_map = char_map.unsqueeze(1).matmul(nb_map).squeeze(1)
            else:
                # char_map_dt = (char_map.detach() * 50).softmax(-1)
                k = min(1 + i * 2, 16)
                char_map_dt = softmax_m1(char_map.detach() * k, dim=-1)
                char_map = char_map_dt.unsqueeze(1).matmul(nb_map).squeeze(1)

        char_maps = torch.stack(char_maps, dim=1)  # [b, L, N], L = n_char + 1
        char_masks = torch.stack(char_masks, dim=1)  # [b, L], 0 denotes masked
        return char_maps, char_masks

    def forward(self, x: torch.FloatTensor, max_char: int = None):
        b, c, h, w = x.size()
        x = x.flatten(2).transpose(1, 2)  # [b, N, c], N = h x w
        g = x.mean(1)  # global representation, [b, c]

        # append eos emb to x
        x_ext = torch.cat(
            [x, self.eos_emb.unsqueeze(0).expand(b, -1).unsqueeze(1)],
            dim=1)  # [b, N+1, c]

        # locate the first character feature
        q_start = self.q_fc(g)  # [b, c]
        k_feat = self.k_fc(x_ext)  # [b, N+1, c]
        start_map = k_feat.matmul(q_start.unsqueeze(-1)).squeeze(
            -1)  # [b, N+1]
        # scaling, referring to: https://kexue.fm/archives/8823
        if self.attn_scaling:
            start_map = start_map / (c**0.5)
        start_map = start_map.softmax(1)

        # Neighbor discovering
        q_feat = self.q_fc(x)
        nb_map = self.neighbor_navigator(q_feat,
                                         k_feat).squeeze(-1)  # [b, N, N+1]
        if self.attn_scaling:
            nb_map = nb_map / (c**0.5)
        nb_map = nb_map.softmax(2)
        last_neighbor = torch.zeros(h * w + 1, device=x.device)
        last_neighbor[-1] = 1.0
        nb_map = torch.cat(
            [
                nb_map,
                last_neighbor.contiguous().view(1, 1, -1).expand(b, -1, -1)
            ],
            dim=1)  # to complete the neighbor matrix, (N+1) x (N+1)

        # string (feature) decoding
        char_maps, char_masks = self.align_chars(start_map, nb_map, max_char)
        char_feats = char_maps.matmul(x_ext)  # [b, L, c]
        char_feats = char_feats * char_masks.unsqueeze(-1)
        logits = self.vis_cls(char_feats)  # [b, L, nC]

        results = dict(
            logits=logits,
            char_feats=char_feats,
            char_maps=char_maps,
            char_masks=char_masks,
            h=h,
            nb_map=nb_map,
        )
        return results


class FeatureMapEnhancer(nn.Module):
    """ Merge the global and local features
    """

    def __init__(self,
                 feat_dim,
                 num_layers=1,
                 focal_level=3,
                 max_kh=1,
                 layerscale_value=1e-6,
                 drop_rate=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(feat_dim)
        self.merge_layer = nn.ModuleList([
            FocalNetBlock(
                dim=feat_dim,
                mlp_ratio=4,
                drop=drop_rate,
                focal_level=focal_level,
                max_kh=max_kh,
                focal_window=3,
                use_layerscale=True,
                layerscale_value=layerscale_value,
            ) for i in range(num_layers)
        ])
        # self.scale = 1. / (feat_dim ** 0.5)
        self.norm2 = nn.LayerNorm(feat_dim)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, feat_map, feat_char, char_attn_map):
        """
        feat_map: [b, N, C]
        feat_char: [b, T, C], T include the EOS token
        char_attn_map: [b, T, N], N exclude the EOS token
        vis_mask: [b, N]
        h: height of the feature map

        return: [b, C, h, w]
        """
        b, C, h, w = feat_map.size()
        feat_map = feat_map.flatten(2).transpose(1, 2)
        # 1. restore the char feats into the visual map
        # char_feat_map = char_attn_map.transpose(1, 2).matmul(feat_char * self.scale) # [b, N, C]
        char_feat_map = char_attn_map.transpose(1, 2).matmul(
            feat_char)  # [b, N, C]
        char_feat_map = self.norm1(char_feat_map)
        feat_map = feat_map + char_feat_map

        # 2. merge
        # vis_mask = vis_mask.contiguous().view(b, h, -1) # [b, h, w]
        for blk in self.merge_layer:
            blk.H, blk.W = h, w
            feat_map = blk(feat_map)
        feat_map = self.dropout(self.norm2(feat_map))
        feat_map = feat_map.transpose(1, 2).reshape(b, C, h, w)  # [b, C, h, w]
        # feat_map = feat_map * vis_mask.unsqueeze(1)
        return feat_map


class LISTERDecoder(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 max_len=25,
                 use_fem=True,
                 detach_grad=False,
                 nhead=8,
                 window_size=11,
                 iters=2,
                 num_sa_layers=1,
                 num_mg_layers=1,
                 coef=[1.0, 0.01, 0.001],
                 **kwargs):
        super().__init__()
        num_classes = out_channels - 1
        self.ignore_index = num_classes
        self.max_len = max_len
        self.use_fem = use_fem
        self.detach_grad = detach_grad
        self.iters = max(1, iters) if use_fem else 0
        feat_dim = in_channels
        self.decoder = NeighborDecoder(num_classes,
                                       feat_dim,
                                       max_len=max_len,
                                       detach_grad=detach_grad,
                                       **kwargs)
        if iters > 0 and use_fem:
            self.cntx_module = LocalAttentionModule(feat_dim,
                                                    nhead,
                                                    window_size,
                                                    num_sa_layers,
                                                    drop_rate=0.1,
                                                    proj_drop_rate=0.1,
                                                    detach_grad=detach_grad,
                                                    mlm=kwargs.get(
                                                        'mlm', False))
            self.merge_layer = FeatureMapEnhancer(feat_dim,
                                                  num_layers=num_mg_layers)
        self.celoss_fn = nn.CrossEntropyLoss(reduction='mean',
                                             ignore_index=self.ignore_index)
        self.coef = coef  # for loss of rec, eos and ent respectively
        # self.coef=(1.0, 0.0, 0.0)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            try:
                nn.init.constant_(m.bias, 0)
            except:
                pass

    def forward(self, x, data=None):
        if data is not None:
            labels, label_lens = data
            label_lens = label_lens + 1
            max_char = label_lens.max()
        else:
            max_char = self.max_len

        res_vis = self.decoder(x, max_char=max_char)
        res_list = [res_vis]
        if self.use_fem:
            for it in range(self.iters):
                char_feat_cntx = self.cntx_module(res_list[-1]['char_feats'],
                                                  res_list[-1]['char_masks'])
                # import ipdb;ipdb.set_trace()
                char_maps = res_list[-1]['char_maps']
                if self.detach_grad:
                    char_maps = char_maps.detach()
                feat_map = self.merge_layer(
                    x,
                    char_feat_cntx,
                    char_maps[:, :, :-1],
                )
                res_i = self.decoder(feat_map, max_char)
                res_list.append(res_i)
        if self.training:
            loss_dict = self.get_loss(res_list[0], labels, label_lens)
            for it in range(self.iters):
                loss_dict_i = self.get_loss(res_list[it + 1], labels,
                                            label_lens)
                for k, v in loss_dict_i.items():
                    loss_dict[k] += v
        else:
            loss_dict = None
        return [loss_dict, res_list[-1]]

    def calc_rec_loss(self, logits, targets):
        """
        Args:
            logits: [minibatch, C, T], not passed to the softmax func.
            targets, torch.cuda.LongTensor [minibatch, T]
            target_lens: [minibatch]
            mask: [minibatch, T]
        """
        losses = self.celoss_fn(logits, targets)
        return losses

    def calc_eos_loc_loss(self, char_maps, target_lens, eps=1e-10):
        max_tok = char_maps.shape[2]
        eos_idx = (target_lens - 1).contiguous().view(-1, 1, 1).expand(
            -1, 1, max_tok)
        eos_maps = torch.gather(char_maps, dim=1,
                                index=eos_idx).squeeze(1)  # (b, max_tok)
        loss = (eos_maps[:, -1] + eps).log().neg()
        return loss.mean()

    def calc_entropy(self, p: torch.Tensor, mask: torch.Tensor, eps=1e-10):
        """
        Args:
            p: probability distribution over the last dimension, of size (..., L, C)
            mask: (..., L)
        """
        p_nlog = (p + eps).log().neg()
        ent = p * p_nlog
        ent = ent.sum(-1) / math.log(p.size(-1) + 1)
        ent = (ent * mask).sum(-1) / (mask.sum(-1) + eps)  # (...)
        ent = ent.mean()
        return ent

    def get_loss(self, model_output, labels, label_lens):
        labels = labels[:, :label_lens.max()]
        batch_size, max_len = labels.size()
        seq_range = torch.arange(
            0, max_len, device=labels.device).long().unsqueeze(0).expand(
                batch_size, max_len)
        seq_len = label_lens.unsqueeze(1).expand_as(seq_range)
        mask = (seq_range < seq_len).float()  # [batch_size, max_len]

        l_rec = self.calc_rec_loss(model_output['logits'].transpose(1, 2),
                                   labels)
        l_eos = self.calc_eos_loc_loss(model_output['char_maps'], label_lens)
        l_ent = self.calc_entropy(model_output['char_maps'], mask)

        loss = l_rec * self.coef[0] + l_eos * self.coef[1] + l_ent * self.coef[
            2]
        loss_dict = dict(
            loss=loss,
            l_rec=l_rec,
            l_eos=l_eos,
            l_ent=l_ent,
        )
        return loss_dict
