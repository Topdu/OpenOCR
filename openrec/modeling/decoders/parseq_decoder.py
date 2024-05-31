# Scene Text Recognition Model Hub
# Copyright 2022 Darwin Bautista
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from itertools import permutations
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules import transformer


class DecoderLayer(nn.Module):
    """A Transformer decoder layer supporting two-stream attention (XLNet) This
    implements a pre-LN decoder, as opposed to the post-LN default in
    PyTorch."""

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation='gelu',
        layer_norm_eps=1e-5,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model,
                                               nhead,
                                               dropout=dropout,
                                               batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model,
                                                nhead,
                                                dropout=dropout,
                                                batch_first=True)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm_q = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm_c = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = transformer._get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.gelu
        super().__setstate__(state)

    def forward_stream(
        self,
        tgt: Tensor,
        tgt_norm: Tensor,
        tgt_kv: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor],
        tgt_key_padding_mask: Optional[Tensor],
    ):
        """Forward pass for a single stream (i.e. content or query) tgt_norm is
        just a LayerNorm'd tgt.

        Added as a separate parameter for efficiency. Both tgt_kv and memory
        are expected to be LayerNorm'd too. memory is LayerNorm'd by ViT.
        """
        tgt2, sa_weights = self.self_attn(
            tgt_norm,
            tgt_kv,
            tgt_kv,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout1(tgt2)

        tgt2, ca_weights = self.cross_attn(self.norm1(tgt), memory, memory)
        self.attn_map = ca_weights
        tgt = tgt + self.dropout2(tgt2)

        tgt2 = self.linear2(
            self.dropout(self.activation(self.linear1(self.norm2(tgt)))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt, sa_weights, ca_weights

    def forward(
        self,
        query,
        content,
        memory,
        query_mask: Optional[Tensor] = None,
        content_mask: Optional[Tensor] = None,
        content_key_padding_mask: Optional[Tensor] = None,
        update_content: bool = True,
    ):
        query_norm = self.norm_q(query)
        content_norm = self.norm_c(content)
        query = self.forward_stream(query, query_norm, content_norm, memory,
                                    query_mask, content_key_padding_mask)[0]
        if update_content:
            content = self.forward_stream(content, content_norm, content_norm,
                                          memory, content_mask,
                                          content_key_padding_mask)[0]
        return query, content


class Decoder(nn.Module):
    __constants__ = ['norm']

    def __init__(self, decoder_layer, num_layers, norm):
        super().__init__()
        self.layers = transformer._get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        query,
        content,
        memory,
        query_mask: Optional[Tensor] = None,
        content_mask: Optional[Tensor] = None,
        content_key_padding_mask: Optional[Tensor] = None,
    ):
        for i, mod in enumerate(self.layers):
            last = i == len(self.layers) - 1
            query, content = mod(
                query,
                content,
                memory,
                query_mask,
                content_mask,
                content_key_padding_mask,
                update_content=not last,
            )
        query = self.norm(query)
        return query


class TokenEmbedding(nn.Module):

    def __init__(self, charset_size: int, embed_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(charset_size, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, tokens: torch.Tensor):
        return math.sqrt(self.embed_dim) * self.embedding(tokens)


class PARSeqDecoder(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 max_label_length=25,
                 embed_dim=384,
                 dec_num_heads=12,
                 dec_mlp_ratio=4,
                 dec_depth=1,
                 perm_num=6,
                 perm_forward=True,
                 perm_mirrored=True,
                 decode_ar=True,
                 refine_iters=1,
                 dropout=0.1,
                 **kwargs: Any) -> None:
        super().__init__()
        self.pad_id = out_channels - 1
        self.eos_id = 0
        self.bos_id = out_channels - 2
        self.max_label_length = max_label_length
        self.decode_ar = decode_ar
        self.refine_iters = refine_iters

        decoder_layer = DecoderLayer(embed_dim, dec_num_heads,
                                     embed_dim * dec_mlp_ratio, dropout)
        self.decoder = Decoder(decoder_layer,
                               num_layers=dec_depth,
                               norm=nn.LayerNorm(embed_dim))

        # Perm/attn mask stuff
        self.rng = np.random.default_rng()
        self.max_gen_perms = perm_num // 2 if perm_mirrored else perm_num
        self.perm_forward = perm_forward
        self.perm_mirrored = perm_mirrored

        # We don't predict <bos> nor <pad>
        self.head = nn.Linear(embed_dim, out_channels - 2)
        self.text_embed = TokenEmbedding(out_channels, embed_dim)

        # +1 for <eos>
        self.pos_queries = nn.Parameter(
            torch.Tensor(1, max_label_length + 1, embed_dim))
        self.dropout = nn.Dropout(p=dropout)
        # Encoder has its own init.
        self.apply(self._init_weights)
        nn.init.trunc_normal_(self.pos_queries, std=0.02)

    def _init_weights(self, module: nn.Module):
        """Initialize the weights using the typical initialization schemes used
        in SOTA models."""

        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight,
                                    mode='fan_out',
                                    nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    @torch.jit.ignore
    def no_weight_decay(self):
        param_names = {'text_embed.embedding.weight', 'pos_queries'}
        return param_names

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[Tensor] = None,
        tgt_padding_mask: Optional[Tensor] = None,
        tgt_query: Optional[Tensor] = None,
        tgt_query_mask: Optional[Tensor] = None,
        pos_query: torch.Tensor = None,
    ):
        N, L = tgt.shape
        # <bos> stands for the null context. We only supply position information for characters after <bos>.
        null_ctx = self.text_embed(tgt[:, :1])

        if tgt_query is None:
            tgt_query = pos_query[:, :L]
        tgt_emb = pos_query[:, :L - 1] + self.text_embed(tgt[:, 1:])
        tgt_emb = self.dropout(torch.cat([null_ctx, tgt_emb], dim=1))

        tgt_query = self.dropout(tgt_query)
        return self.decoder(tgt_query, tgt_emb, memory, tgt_query_mask,
                            tgt_mask, tgt_padding_mask)

    def forward(self, x, data=None, pos_query=None):
        if self.training:
            return self.training_step([x, pos_query, data[0]])
        else:
            return self.forward_test(x, pos_query)

    def forward_test(self,
                     memory: Tensor,
                     pos_query: Tensor = None,
                     max_length: Optional[int] = None) -> Tensor:
        _device = memory.get_device()
        testing = max_length is None
        max_length = (self.max_label_length if max_length is None else min(
            max_length, self.max_label_length))
        bs = memory.shape[0]
        # +1 for <eos> at end of sequence.
        num_steps = max_length + 1
        # memory = self.encode(images)

        # Query positions up to `num_steps`
        if pos_query is None:
            pos_queries = self.pos_queries[:, :num_steps].expand(bs, -1, -1)
        else:
            pos_queries = pos_query

        # Special case for the forward permutation. Faster than using `generate_attn_masks()`
        tgt_mask = query_mask = torch.triu(
            torch.full((num_steps, num_steps), float('-inf'), device=_device),
            1)
        self.attn_maps = []
        if self.decode_ar:
            tgt_in = torch.full((bs, num_steps),
                                self.pad_id,
                                dtype=torch.long,
                                device=_device)
            tgt_in[:, 0] = self.bos_id

            logits = []
            for i in range(num_steps):
                j = i + 1  # next token index
                # Efficient decoding:
                # Input the context up to the ith token. We use only one query (at position = i) at a time.
                # This works because of the lookahead masking effect of the canonical (forward) AR context.
                # Past tokens have no access to future tokens, hence are fixed once computed.
                tgt_out = self.decode(
                    tgt_in[:, :j],
                    memory,
                    tgt_mask[:j, :j],
                    tgt_query=pos_queries[:, i:j],
                    tgt_query_mask=query_mask[i:j, :j],
                    pos_query=pos_queries,
                )
                self.attn_maps.append(self.decoder.layers[-1].attn_map)
                # the next token probability is in the output's ith token position
                p_i = self.head(tgt_out)
                logits.append(p_i)
                if j < num_steps:
                    # greedy decode. add the next token index to the target input
                    tgt_in[:, j] = p_i.squeeze().argmax(-1)
                    # Efficient batch decoding: If all output words have at least one EOS token, end decoding.
                    if testing and (tgt_in == self.eos_id).any(dim=-1).all():
                        break

            logits = torch.cat(logits, dim=1)
        else:
            # No prior context, so input is just <bos>. We query all positions.
            tgt_in = torch.full((bs, 1),
                                self.bos_id,
                                dtype=torch.long,
                                device=_device)
            tgt_out = self.decode(tgt_in,
                                  memory,
                                  tgt_query=pos_queries,
                                  pos_query=pos_queries)
            logits = self.head(tgt_out)

        if self.refine_iters:
            # For iterative refinement, we always use a 'cloze' mask.
            # We can derive it from the AR forward mask by unmasking the token context to the right.
            query_mask[torch.triu(
                torch.ones(num_steps,
                           num_steps,
                           dtype=torch.bool,
                           device=_device), 2)] = 0
            bos = torch.full((bs, 1),
                             self.bos_id,
                             dtype=torch.long,
                             device=_device)
            for i in range(self.refine_iters):
                # Prior context is the previous output.
                tgt_in = torch.cat([bos, logits[:, :-1].argmax(-1)], dim=1)
                tgt_padding_mask = (tgt_in == self.eos_id).int().cumsum(
                    -1) > 0  # mask tokens beyond the first EOS token.
                tgt_out = self.decode(
                    tgt_in,
                    memory,
                    tgt_mask,
                    tgt_padding_mask,
                    tgt_query=pos_queries,
                    tgt_query_mask=query_mask[:, :tgt_in.shape[1]],
                    pos_query=pos_queries,
                )
                logits = self.head(tgt_out)

        return F.softmax(logits, -1)

    def gen_tgt_perms(self, tgt, _device):
        """Generate shared permutations for the whole batch.

        This works because the same attention mask can be used for the shorter
        sequences because of the padding mask.
        """
        # We don't permute the position of BOS, we permute EOS separately
        max_num_chars = tgt.shape[1] - 2
        # Special handling for 1-character sequences
        if max_num_chars == 1:
            return torch.arange(3, device=_device).unsqueeze(0)
        perms = [torch.arange(max_num_chars, device=_device)
                 ] if self.perm_forward else []
        # Additional permutations if needed
        max_perms = math.factorial(max_num_chars)
        if self.perm_mirrored:
            max_perms //= 2
        num_gen_perms = min(self.max_gen_perms, max_perms)
        # For 4-char sequences and shorter, we generate all permutations and sample from the pool to avoid collisions
        # Note that this code path might NEVER get executed since the labels in a mini-batch typically exceed 4 chars.
        if max_num_chars < 5:
            # Pool of permutations to sample from. We only need the first half (if complementary option is selected)
            # Special handling for max_num_chars == 4 which correctly divides the pool into the flipped halves
            if max_num_chars == 4 and self.perm_mirrored:
                selector = [0, 3, 4, 6, 9, 10, 12, 16, 17, 18, 19, 21]
            else:
                selector = list(range(max_perms))
            perm_pool = torch.as_tensor(list(
                permutations(range(max_num_chars), max_num_chars)),
                                        device=_device)[selector]
            # If the forward permutation is always selected, no need to add it to the pool for sampling
            if self.perm_forward:
                perm_pool = perm_pool[1:]
            perms = torch.stack(perms)
            if len(perm_pool):
                i = self.rng.choice(len(perm_pool),
                                    size=num_gen_perms - len(perms),
                                    replace=False)
                perms = torch.cat([perms, perm_pool[i]])
        else:
            perms.extend([
                torch.randperm(max_num_chars, device=_device)
                for _ in range(num_gen_perms - len(perms))
            ])
            perms = torch.stack(perms)
        if self.perm_mirrored:
            # Add complementary pairs
            comp = perms.flip(-1)
            # Stack in such a way that the pairs are next to each other.
            perms = torch.stack([perms, comp
                                 ]).transpose(0, 1).reshape(-1, max_num_chars)
        # NOTE:
        # The only meaningful way of permuting the EOS position is by moving it one character position at a time.
        # However, since the number of permutations = T! and number of EOS positions = T + 1, the number of possible EOS
        # positions will always be much less than the number of permutations (unless a low perm_num is set).
        # Thus, it would be simpler to just train EOS using the full and null contexts rather than trying to evenly
        # distribute it across the chosen number of permutations.
        # Add position indices of BOS and EOS
        bos_idx = perms.new_zeros((len(perms), 1))
        eos_idx = perms.new_full((len(perms), 1), max_num_chars + 1)
        perms = torch.cat([bos_idx, perms + 1, eos_idx], dim=1)
        # Special handling for the reverse direction. This does two things:
        # 1. Reverse context for the characters
        # 2. Null context for [EOS] (required for learning to predict [EOS] in NAR mode)
        if len(perms) > 1:
            perms[1, 1:] = max_num_chars + 1 - torch.arange(max_num_chars + 1,
                                                            device=_device)
        return perms

    def generate_attn_masks(self, perm, _device):
        """Generate attention masks given a sequence permutation (includes pos.
        for bos and eos tokens)

        :param perm: the permutation sequence. i = 0 is always the BOS
        :return: lookahead attention masks
        """
        sz = perm.shape[0]
        mask = torch.zeros((sz, sz), device=_device)
        for i in range(sz):
            query_idx = perm[i]
            masked_keys = perm[i + 1:]
            mask[query_idx, masked_keys] = float('-inf')
        content_mask = mask[:-1, :-1].clone()
        mask[torch.eye(sz, dtype=torch.bool,
                       device=_device)] = float('-inf')  # mask "self"
        query_mask = mask[1:, :-1]
        return content_mask, query_mask

    def training_step(self, batch):
        memory, pos_query, tgt = batch
        bs = memory.shape[0]
        if pos_query is None:
            pos_query = self.pos_queries.expand(bs, -1, -1)

        # Prepare the target sequences (input and output)
        tgt_perms = self.gen_tgt_perms(tgt, memory.get_device())
        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:]
        # The [EOS] token is not depended upon by any other token in any permutation ordering
        tgt_padding_mask = (tgt_in == self.pad_id) | (tgt_in == self.eos_id)

        loss = 0
        loss_numel = 0
        n = (tgt_out != self.pad_id).sum().item()
        for i, perm in enumerate(tgt_perms):
            tgt_mask, query_mask = self.generate_attn_masks(
                perm, memory.get_device())
            out = self.decode(
                tgt_in,
                memory,
                tgt_mask,
                tgt_padding_mask,
                tgt_query_mask=query_mask,
                pos_query=pos_query,
            )
            logits = self.head(out)
            if i == 0:
                final_out = logits
            loss += n * F.cross_entropy(logits.flatten(end_dim=1),
                                        tgt_out.flatten(),
                                        ignore_index=self.pad_id)
            loss_numel += n
            # After the second iteration (i.e. done with canonical and reverse orderings),
            # remove the [EOS] tokens for the succeeding perms
            if i == 1:
                tgt_out = torch.where(tgt_out == self.eos_id, self.pad_id,
                                      tgt_out)
                n = (tgt_out != self.pad_id).sum().item()
        loss /= loss_numel

        # self.log('loss', loss)
        return [loss, final_out]
