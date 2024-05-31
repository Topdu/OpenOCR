import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_, ones_, trunc_normal_, zeros_

from openrec.modeling.common import DropPath, Identity, Mlp


class ConvBNLayer(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=0,
        bias=False,
        groups=1,
        act=nn.GELU,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias,
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = act()

    def forward(self, inputs):
        out = self.conv(inputs)
        out = self.norm(out)
        out = self.act(out)
        return out


class ConvMixer(nn.Module):

    def __init__(
        self,
        dim,
        num_heads=8,
        local_k=[5, 5],
    ):
        super().__init__()
        self.local_mixer = nn.Conv2d(dim, dim, 5, 1, 2, groups=num_heads)

    def forward(self, x, mask=None):
        x = self.local_mixer(x)
        return x


class ConvMlp(nn.Module):

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
        groups=1,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, groups=groups)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        B, N, _ = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = q @ k.transpose(-2, -1) * self.scale
        if mask is not None:
            attn += mask.unsqueeze(0)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, self.dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(
        self,
        dim,
        num_heads,
        mixer='Global',
        local_k=[7, 11],
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        eps=1e-6,
    ):
        super().__init__()
        mlp_hidden_dim = int(dim * mlp_ratio)
        if mixer == 'Global' or mixer == 'Local':
            self.norm1 = norm_layer(dim, eps=eps)
            self.mixer = Attention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
            )
            self.norm2 = norm_layer(dim, eps=eps)
            self.mlp = Mlp(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer,
                drop=drop,
            )
        elif mixer == 'Conv':
            self.norm1 = nn.BatchNorm2d(dim)
            self.mixer = ConvMixer(dim, num_heads=num_heads, local_k=local_k)
            self.norm2 = nn.BatchNorm2d(dim)
            self.mlp = ConvMlp(in_features=dim,
                               hidden_features=mlp_hidden_dim,
                               act_layer=act_layer,
                               drop=drop)
        else:
            raise TypeError('The mixer must be one of [Global, Local, Conv]')

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else Identity()

    def forward(self, x, mask=None):
        x = self.norm1(x + self.drop_path(self.mixer(x, mask=mask)))
        x = self.norm2(x + self.drop_path(self.mlp(x)))
        return x


class FlattenTranspose(nn.Module):

    def forward(self, x, mask=None):
        return x.flatten(2).transpose(1, 2)


class SVTRStage(nn.Module):

    def __init__(self,
                 feat_maxSize=[16, 128],
                 dim=64,
                 out_dim=256,
                 depth=3,
                 mixer=['Local'] * 3,
                 local_k=[7, 11],
                 sub_k=[2, 1],
                 num_heads=2,
                 mlp_ratio=4,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.0,
                 attn_drop_rate=0.0,
                 drop_path=[0.1] * 3,
                 norm_layer=nn.LayerNorm,
                 act=nn.GELU,
                 eps=1e-6,
                 downsample=None,
                 **kwargs):
        super().__init__()
        self.dim = dim

        conv_block_num = sum([1 if mix == 'Conv' else 0 for mix in mixer])
        if conv_block_num == depth:
            self.mask = None
            conv_block_num = 0
            if downsample:
                self.sub_norm = nn.BatchNorm2d(out_dim, eps=eps)
        else:
            if 'Local' in mixer:
                mask = self.get_max2d_mask(feat_maxSize[0], feat_maxSize[1],
                                           local_k)
                self.register_buffer('mask', mask)
            else:
                self.mask = None
            if downsample:
                self.sub_norm = norm_layer(out_dim, eps=eps)
        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(
                Block(
                    dim=dim,
                    num_heads=num_heads,
                    mixer=mixer[i],
                    local_k=local_k,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    act_layer=act,
                    attn_drop=attn_drop_rate,
                    drop_path=drop_path[i],
                    norm_layer=norm_layer,
                    eps=eps,
                ))
            if i == conv_block_num - 1:
                self.blocks.append(FlattenTranspose())

        if downsample:
            self.downsample = nn.Conv2d(dim,
                                        out_dim,
                                        kernel_size=3,
                                        stride=sub_k,
                                        padding=1)
        else:
            self.downsample = None

    def get_max2d_mask(self, H, W, local_k):
        hk, wk = local_k
        mask = torch.ones(H * W,
                          H + hk - 1,
                          W + wk - 1,
                          dtype=torch.float32,
                          requires_grad=False)
        for h in range(0, H):
            for w in range(0, W):
                mask[h * W + w, h:h + hk, w:w + wk] = 0.0
        mask = mask[:, hk // 2:H + hk // 2, wk // 2:W + wk // 2]  # .flatten(1)
        mask[mask >= 1] = -np.inf
        return mask.reshape(H, W, H, W)

    def get_2d_mask(self, H1, W1):

        if H1 == self.mask.shape[0] and W1 == self.mask.shape[1]:
            return self.mask.flatten(0, 1).flatten(1, 2).unsqueeze(0)
        h_slice = H1 // 2
        offet_h = H1 - 2 * h_slice
        w_slice = W1 // 2
        offet_w = W1 - 2 * w_slice
        mask1 = self.mask[:h_slice + offet_h, :w_slice, :H1, :W1]
        mask2 = self.mask[:h_slice + offet_h, -w_slice:, :H1, -W1:]
        mask3 = self.mask[-h_slice:, :(w_slice + offet_w), -H1:, :W1]
        mask4 = self.mask[-h_slice:, -(w_slice + offet_w):, -H1:, -W1:]

        mask_top = torch.concat([mask1, mask2], 1)
        mask_bott = torch.concat([mask3, mask4], 1)
        mask = torch.concat([mask_top.flatten(2), mask_bott.flatten(2)], 0)
        return mask.flatten(0, 1).unsqueeze(0)

    def forward(self, x, sz=None):
        if self.mask is not None:
            mask = self.get_2d_mask(sz[0], sz[1])
        else:
            mask = self.mask
        for blk in self.blocks:
            x = blk(x, mask=mask)

        if self.downsample is not None:
            if x.dim() == 3:
                x = x.transpose(1, 2).reshape(-1, self.dim, sz[0], sz[1])
                x = self.downsample(x)
                sz = x.shape[2:]
                x = x.flatten(2).transpose(1, 2)
            else:
                x = self.downsample(x)
                sz = x.shape[2:]
            x = self.sub_norm(x)
        return x, sz


class POPatchEmbed(nn.Module):
    """Image to Patch Embedding."""

    def __init__(self,
                 in_channels=3,
                 feat_max_size=[8, 32],
                 embed_dim=768,
                 use_pos_embed=False,
                 flatten=False):
        super().__init__()
        self.patch_embed = nn.Sequential(
            ConvBNLayer(
                in_channels=in_channels,
                out_channels=embed_dim // 2,
                kernel_size=3,
                stride=2,
                padding=1,
                act=nn.GELU,
                bias=None,
            ),
            ConvBNLayer(
                in_channels=embed_dim // 2,
                out_channels=embed_dim,
                kernel_size=3,
                stride=2,
                padding=1,
                act=nn.GELU,
                bias=None,
            ),
        )
        self.use_pos_embed = use_pos_embed
        self.flatten = flatten
        if use_pos_embed:
            pos_embed = torch.zeros(
                [1, feat_max_size[0] * feat_max_size[1], embed_dim],
                dtype=torch.float32)
            trunc_normal_(pos_embed, mean=0, std=0.02)
            self.pos_embed = nn.Parameter(
                pos_embed.transpose(1,
                                    2).reshape(1, embed_dim, feat_max_size[0],
                                               feat_max_size[1]),
                requires_grad=True,
            )

    def forward(self, x):
        x = self.patch_embed(x)
        sz = x.shape[2:]
        if self.use_pos_embed:
            x = x + self.pos_embed[:, :, :sz[0], :sz[1]]
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)
        return x, sz


class SVTRv2(nn.Module):

    def __init__(self,
                 max_sz=[32, 128],
                 in_channels=3,
                 out_channels=192,
                 depths=[3, 6, 3],
                 dims=[64, 128, 256],
                 mixer=[['Local'] * 3, ['Local'] * 3 + ['Global'] * 3,
                        ['Global'] * 3],
                 use_pos_embed=True,
                 local_k=[[7, 11], [7, 11], [-1, -1]],
                 sub_k=[[1, 1], [2, 1], [1, 1]],
                 num_heads=[2, 4, 8],
                 mlp_ratio=4,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.0,
                 last_drop=0.1,
                 attn_drop_rate=0.0,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 act=nn.GELU,
                 last_stage=False,
                 eps=1e-6,
                 **kwargs):
        super().__init__()
        num_stages = len(depths)
        self.num_features = dims[-1]

        feat_max_size = [max_sz[0] // 4, max_sz[1] // 4]
        self.pope = POPatchEmbed(in_channels=in_channels,
                                 feat_max_size=feat_max_size,
                                 embed_dim=dims[0],
                                 use_pos_embed=use_pos_embed,
                                 flatten=mixer[0][0] != 'Conv')

        dpr = np.linspace(0, drop_path_rate,
                          sum(depths))  # stochastic depth decay rule

        self.stages = nn.ModuleList()
        for i_stage in range(num_stages):
            stage = SVTRStage(
                feat_maxSize=feat_max_size,
                dim=dims[i_stage],
                out_dim=dims[i_stage + 1] if i_stage < num_stages - 1 else 0,
                depth=depths[i_stage],
                mixer=mixer[i_stage],
                local_k=local_k[i_stage],
                sub_k=sub_k[i_stage],
                num_heads=num_heads[i_stage],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_stage]):sum(depths[:i_stage + 1])],
                norm_layer=norm_layer,
                act=act,
                downsample=False if i_stage == num_stages - 1 else True,
                eps=eps,
            )
            self.stages.append(stage)
            feat_max_size = [
                feat_max_size[0] // sub_k[i_stage][0],
                feat_max_size[1] // sub_k[i_stage][1]
            ]

        self.out_channels = self.num_features
        self.last_stage = last_stage
        if last_stage:
            self.out_channels = out_channels
            self.last_conv = nn.Linear(self.num_features,
                                       self.out_channels,
                                       bias=False)
            self.hardswish = nn.Hardswish()
            self.dropout = nn.Dropout(p=last_drop)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
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
        return {'patch_embed', 'downsample', 'pos_embed'}

    def forward(self, x):
        x, sz = self.pope(x)

        for stage in self.stages:
            x, sz = stage(x, sz)

        if self.last_stage:
            x = x.reshape(-1, sz[0], sz[1], self.num_features)
            x = x.mean(1)
            x = self.last_conv(x)
            x = self.hardswish(x)
            x = self.dropout(x)

        return x
