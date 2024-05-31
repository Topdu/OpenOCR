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
from torch.nn.init import trunc_normal_


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

    def __init__(self, num_classes, feat_dim, max_len=1000, **kwargs):
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

        self.detach_grad = kwargs['detach_grad']
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


class LISTERDecoder(nn.Module):

    def __init__(self, in_channels, out_channels, max_len=25, **kwargs):
        super().__init__()
        num_classes = out_channels - 1
        self.ignore_index = num_classes
        self.max_len = max_len
        self.decoder = NeighborDecoder(num_classes,
                                       in_channels,
                                       max_len=max_len,
                                       **kwargs)
        self.celoss_fn = nn.CrossEntropyLoss(reduction='mean',
                                             ignore_index=self.ignore_index)
        self.coef = (1.0, 0.01, 0.001
                     )  # for loss of rec, eos and ent respectively
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
        ret = self.decoder(x, max_char=max_char)
        if self.training:
            loss_dict = self.get_loss(ret, labels, label_lens)
        else:
            loss_dict = None
        return [loss_dict, ret]

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
