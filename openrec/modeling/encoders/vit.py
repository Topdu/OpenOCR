import numpy as np
import torch
from torch import nn
from torch.nn.init import kaiming_normal_, ones_, trunc_normal_, zeros_

from openrec.modeling.common import Block, PatchEmbed
from openrec.modeling.encoders.svtrv2_lnconv import Feat2D, LastStage


class ViT(nn.Module):

    def __init__(
        self,
        img_size=[32, 128],
        patch_size=[4, 8],
        in_channels=3,
        out_channels=256,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        last_stage=False,
        feat2d=False,
        use_cls_token=False,
        **kwargs,
    ):
        super().__init__()
        self.img_size = img_size
        self.embed_dim = embed_dim
        self.out_channels = embed_dim
        self.use_cls_token = use_cls_token
        self.feat_sz = [
            img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        ]

        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels,
                                      embed_dim)
        num_patches = self.patch_embed.num_patches
        if use_cls_token:
            self.cls_token = nn.Parameter(
                torch.zeros([1, 1, embed_dim], dtype=torch.float32),
                requires_grad=True,
            )
            trunc_normal_(self.cls_token, mean=0, std=0.02)
            self.pos_embed = nn.Parameter(
                torch.zeros([1, num_patches + 1, embed_dim],
                            dtype=torch.float32),
                requires_grad=True,
            )
        else:
            self.pos_embed = nn.Parameter(
                torch.zeros([1, num_patches, embed_dim], dtype=torch.float32),
                requires_grad=True,
            )
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = np.linspace(0, drop_path_rate, depth)
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                act_layer=act_layer,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
            ) for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
        self.last_stage = last_stage
        self.feat2d = feat2d
        if last_stage:
            self.out_channels = out_channels
            self.stages = LastStage(embed_dim, out_channels, last_drop=0.1)
        if feat2d:
            self.stages = Feat2D()
        trunc_normal_(self.pos_embed, mean=0, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, mean=0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                zeros_(m.bias)
        if isinstance(m, nn.LayerNorm):
            zeros_(m.bias)
            ones_(m.weight)
        if isinstance(m, nn.Conv2d):
            kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed'}

    def forward(self, x):
        x = self.patch_embed(x)
        if self.use_cls_token:
            x = torch.concat([self.cls_token.tile([x.shape[0], 1, 1]), x], 1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        if self.use_cls_token:
            x = x[:, 1:, :]
        if self.last_stage:
            x, sz = self.stages(x, self.feat_sz)
        if self.feat2d:
            x, sz = self.stages(x, self.feat_sz)
        return x
