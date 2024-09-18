'''
This code is refer from:
https://github.com/AlibabaResearch/AdvancedLiterateMachinery/blob/main/OCR/MGP-STR
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenLearner(nn.Module):

    def __init__(self, input_embed_dim, out_token=30):
        super().__init__()
        self.token_norm = nn.LayerNorm(input_embed_dim)
        self.tokenLearner = nn.Sequential(
            nn.Conv2d(input_embed_dim,
                      input_embed_dim,
                      kernel_size=(1, 1),
                      stride=1,
                      groups=8,
                      bias=False),
            nn.Conv2d(input_embed_dim,
                      out_token,
                      kernel_size=(1, 1),
                      stride=1,
                      bias=False))
        self.feat = nn.Conv2d(input_embed_dim,
                              input_embed_dim,
                              kernel_size=(1, 1),
                              stride=1,
                              groups=8,
                              bias=False)
        self.norm = nn.LayerNorm(input_embed_dim)

    def forward(self, x):
        x = self.token_norm(x)  # [bs, 257, 768]
        x = x.transpose(1, 2).unsqueeze(-1)  # [bs, 768, 257, 1]
        selected = self.tokenLearner(x)  # [bs, 27, 257, 1].
        selected = selected.flatten(2)  # [bs, 27, 257].
        selected = F.softmax(selected, dim=-1)
        feat = self.feat(x)  #  [bs, 768, 257, 1].
        feat = feat.flatten(2).transpose(1, 2)  # [bs, 257, 768]
        x = torch.einsum('...si,...id->...sd', selected, feat)  # [bs, 27, 768]

        x = self.norm(x)
        return selected, x


class MGPDecoder(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 max_len=25,
                 only_char=False,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        num_classes = out_channels
        embed_dim = in_channels
        self.batch_max_length = max_len + 2
        self.char_tokenLearner = TokenLearner(embed_dim, self.batch_max_length)
        self.char_head = nn.Linear(
            embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.only_char = only_char
        if not only_char:
            self.bpe_tokenLearner = TokenLearner(embed_dim,
                                                 self.batch_max_length)
            self.wp_tokenLearner = TokenLearner(embed_dim,
                                                self.batch_max_length)
            self.bpe_head = nn.Linear(
                embed_dim, 50257) if num_classes > 0 else nn.Identity()
            self.wp_head = nn.Linear(
                embed_dim, 30522) if num_classes > 0 else nn.Identity()

    def forward(self, x, data=None):
        # attens = []
        # char
        char_attn, x_char = self.char_tokenLearner(x)
        x_char = self.char_head(x_char)
        char_out = x_char
        # attens = [char_attn]
        if not self.only_char:
            # bpe
            bpe_attn, x_bpe = self.bpe_tokenLearner(x)
            bpe_out = self.bpe_head(x_bpe)
            # attens += [bpe_attn]
            # wp
            wp_attn, x_wp = self.wp_tokenLearner(x)
            wp_out = self.wp_head(x_wp)
            return [char_out, bpe_out, wp_out] if self.training else [
                F.softmax(char_out, -1),
                F.softmax(bpe_out, -1),
                F.softmax(wp_out, -1)
            ]
            # attens += [wp_attn]

        return char_out if self.training else F.softmax(char_out, -1)
