# --------------------------------------------------------
# FocalNets -- Focal Modulation Networks
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Jianwei Yang (jianwyan@microsoft.com)
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from torch.nn.init import trunc_normal_

from openrec.modeling.common import DropPath, Mlp
from openrec.modeling.encoders.svtrnet import ConvBNLayer


class FocalModulation(nn.Module):

    def __init__(self,
                 dim,
                 focal_window,
                 focal_level,
                 max_kh=None,
                 focal_factor=2,
                 bias=True,
                 proj_drop=0.0,
                 use_postln_in_modulation=False,
                 normalize_modulator=False):
        super().__init__()

        self.dim = dim
        self.focal_window = focal_window
        self.focal_level = focal_level
        self.focal_factor = focal_factor
        self.use_postln_in_modulation = use_postln_in_modulation
        self.normalize_modulator = normalize_modulator

        self.f = nn.Linear(dim, 2 * dim + (self.focal_level + 1), bias=bias)
        self.h = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=bias)

        self.act = nn.GELU()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.focal_layers = nn.ModuleList()

        self.kernel_sizes = []
        for k in range(self.focal_level):
            kernel_size = self.focal_factor * k + self.focal_window
            if max_kh is not None:
                k_h, k_w = [min(kernel_size, max_kh), kernel_size]
                kernel_size = [k_h, k_w]
                padding = [k_h // 2, k_w // 2]
            else:
                padding = kernel_size // 2
            self.focal_layers.append(
                nn.Sequential(
                    nn.Conv2d(dim,
                              dim,
                              kernel_size=kernel_size,
                              stride=1,
                              groups=dim,
                              padding=padding,
                              bias=False),
                    nn.GELU(),
                ))
            self.kernel_sizes.append(kernel_size)
        if self.use_postln_in_modulation:
            self.ln = nn.LayerNorm(dim)

    def forward(self, x):
        """
        Args:
            x: input features with shape of (B, H, W, C)
        """
        C = x.shape[-1]

        # pre linear projection
        x = self.f(x).permute(0, 3, 1, 2).contiguous()
        q, ctx, self.gates = torch.split(x, (C, C, self.focal_level + 1), 1)

        # context aggreation
        ctx_all = 0
        for l in range(self.focal_level):
            ctx = self.focal_layers[l](ctx)
            ctx_all = ctx_all + ctx * self.gates[:, l:l + 1]
        ctx_global = self.act(ctx.mean(2, keepdim=True).mean(3, keepdim=True))
        ctx_all = ctx_all + ctx_global * self.gates[:, self.focal_level:]

        # normalize context
        if self.normalize_modulator:
            ctx_all = ctx_all / (self.focal_level + 1)

        # focal modulation
        self.modulator = self.h(ctx_all)
        x_out = q * self.modulator
        x_out = x_out.permute(0, 2, 3, 1).contiguous()
        if self.use_postln_in_modulation:
            x_out = self.ln(x_out)

        # post linear porjection
        x_out = self.proj(x_out)
        x_out = self.proj_drop(x_out)
        return x_out

    def extra_repr(self) -> str:
        return f'dim={self.dim}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0

        flops += N * self.dim * (self.dim * 2 + (self.focal_level + 1))

        # focal convolution
        for k in range(self.focal_level):
            flops += N * (self.kernel_sizes[k]**2 + 1) * self.dim

        # global gating
        flops += N * 1 * self.dim

        #  self.linear
        flops += N * self.dim * (self.dim + 1)

        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class FocalNetBlock(nn.Module):
    r"""Focal Modulation Network Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        focal_level (int): Number of focal levels.
        focal_window (int): Focal window size at first focal level
        use_layerscale (bool): Whether use layerscale
        layerscale_value (float): Initial layerscale value
        use_postln (bool): Whether use layernorm after modulation
    """

    def __init__(
        self,
        dim,
        input_resolution=None,
        mlp_ratio=4.0,
        drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        focal_level=1,
        focal_window=3,
        max_kh=None,
        use_layerscale=False,
        layerscale_value=1e-4,
        use_postln=False,
        use_postln_in_modulation=False,
        normalize_modulator=False,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.mlp_ratio = mlp_ratio

        self.focal_window = focal_window
        self.focal_level = focal_level
        self.use_postln = use_postln

        self.norm1 = norm_layer(dim)
        self.modulation = FocalModulation(
            dim,
            proj_drop=drop,
            focal_window=focal_window,
            focal_level=self.focal_level,
            max_kh=max_kh,
            use_postln_in_modulation=use_postln_in_modulation,
            normalize_modulator=normalize_modulator,
        )

        self.drop_path = DropPath(
            drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer,
                       drop=drop)

        self.gamma_1 = 1.0
        self.gamma_2 = 1.0
        if use_layerscale:
            self.gamma_1 = nn.Parameter(layerscale_value * torch.ones((dim)),
                                        requires_grad=True)
            self.gamma_2 = nn.Parameter(layerscale_value * torch.ones((dim)),
                                        requires_grad=True)

        self.H = None
        self.W = None

    def forward(self, x):
        H, W = self.H, self.W
        B, L, C = x.shape
        shortcut = x

        # Focal Modulation
        x = x if self.use_postln else self.norm1(x)
        x = x.view(B, H, W, C)
        x = self.modulation(x).view(B, H * W, C)
        x = x if not self.use_postln else self.norm1(x)

        # FFN
        x = shortcut + self.drop_path(self.gamma_1 * x)
        x = x + self.drop_path(self.gamma_2 * (self.norm2(
            self.mlp(x)) if self.use_postln else self.mlp(self.norm2(x))))

        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, input_resolution={self.input_resolution}, ' f'mlp_ratio={self.mlp_ratio}'

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W

        # W-MSA/SW-MSA
        flops += self.modulation.flops(H * W)

        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class BasicLayer(nn.Module):
    """A basic Focal Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        focal_level (int): Number of focal levels
        focal_window (int): Focal window size at first focal level
        use_layerscale (bool): Whether use layerscale
        layerscale_value (float): Initial layerscale value
        use_postln (bool): Whether use layernorm after modulation
    """

    def __init__(
        self,
        dim,
        out_dim,
        input_resolution,
        depth,
        mlp_ratio=4.0,
        drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        downsample=None,
        downsample_kernel=[],
        use_checkpoint=False,
        focal_level=1,
        focal_window=1,
        use_conv_embed=False,
        use_layerscale=False,
        layerscale_value=1e-4,
        use_postln=False,
        use_postln_in_modulation=False,
        normalize_modulator=False,
    ):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            FocalNetBlock(
                dim=dim,
                input_resolution=input_resolution,
                mlp_ratio=mlp_ratio,
                drop=drop,
                drop_path=drop_path[i]
                if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                focal_level=focal_level,
                focal_window=focal_window,
                use_layerscale=use_layerscale,
                layerscale_value=layerscale_value,
                use_postln=use_postln,
                use_postln_in_modulation=use_postln_in_modulation,
                normalize_modulator=normalize_modulator,
            ) for i in range(depth)
        ])

        if downsample is not None:
            self.downsample = downsample(
                img_size=input_resolution,
                patch_size=downsample_kernel,
                in_chans=dim,
                embed_dim=out_dim,
                use_conv_embed=use_conv_embed,
                norm_layer=norm_layer,
                is_stem=False,
            )
        else:
            self.downsample = None

    def forward(self, x, H, W):
        for blk in self.blocks:
            blk.H, blk.W = H, W
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)

        if self.downsample is not None:
            x = x.transpose(1, 2).reshape(x.shape[0], -1, H, W)
            x, Ho, Wo = self.downsample(x)
        else:
            Ho, Wo = H, W
        return x, Ho, Wo

    def extra_repr(self) -> str:
        return f'dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}'

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class PatchEmbed(nn.Module):
    r"""Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self,
                 img_size=(224, 224),
                 patch_size=[4, 4],
                 in_chans=3,
                 embed_dim=96,
                 use_conv_embed=False,
                 norm_layer=None,
                 is_stem=False):
        super().__init__()
        # patch_size = to_2tuple(patch_size)
        patches_resolution = [
            img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        ]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if use_conv_embed:
            # if we choose to use conv embedding, then we treat the stem and non-stem differently
            if is_stem:
                kernel_size = 7
                padding = 2
                stride = 4
            else:
                kernel_size = 3
                padding = 1
                stride = 2
            self.proj = nn.Conv2d(in_chans,
                                  embed_dim,
                                  kernel_size=kernel_size,
                                  stride=stride,
                                  padding=padding)
        else:
            self.proj = nn.Conv2d(in_chans,
                                  embed_dim,
                                  kernel_size=patch_size,
                                  stride=patch_size)

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape

        x = self.proj(x)
        H, W = x.shape[2:]
        x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x, H, W

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (
            self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class FocalSVTR(nn.Module):
    r"""Focal Modulation Networks (FocalNets)

    Args:
        img_size (int | tuple(int)): Input image size. Default [32, 128]
        patch_size (int | tuple(int)): Patch size. Default: [4, 4]
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Focal Transformer layer.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        drop_rate (float): Dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        focal_levels (list): How many focal levels at all stages. Note that this excludes the finest-grain level. Default: [1, 1, 1, 1]
        focal_windows (list): The focal window size at all stages. Default: [7, 5, 3, 1]
        use_conv_embed (bool): Whether use convolutional embedding. We noted that using convolutional embedding usually improve the performance,
        but we do not use it by default. Default: False
        use_layerscale (bool): Whether use layerscale proposed in CaiT. Default: False
        layerscale_value (float): Value for layer scale. Default: 1e-4
        use_postln (bool): Whether use layernorm after modulation (it helps stablize training of large models)
    """

    def __init__(
        self,
        img_size=[32, 128],
        patch_size=[4, 4],
        out_channels=256,
        out_char_num=25,
        in_channels=3,
        embed_dim=96,
        depths=[3, 6, 3],
        sub_k=[[2, 1], [2, 1], [1, 1]],
        last_stage=False,
        mlp_ratio=4.0,
        drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        patch_norm=True,
        use_checkpoint=False,
        focal_levels=[6, 6, 6],
        focal_windows=[3, 3, 3],
        use_conv_embed=False,
        use_layerscale=False,
        layerscale_value=1e-4,
        use_postln=False,
        use_postln_in_modulation=False,
        normalize_modulator=False,
        feat2d=False,
        **kwargs,
    ):
        super().__init__()

        self.num_layers = len(depths)
        embed_dim = [embed_dim * (2**i) for i in range(self.num_layers)]
        self.feat2d = feat2d
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = embed_dim[-1]
        self.mlp_ratio = mlp_ratio

        self.patch_embed = nn.Sequential(
            ConvBNLayer(
                in_channels=in_channels,
                out_channels=embed_dim[0] // 2,
                kernel_size=3,
                stride=2,
                padding=1,
                act=nn.GELU,
                bias=None,
            ),
            ConvBNLayer(
                in_channels=embed_dim[0] // 2,
                out_channels=embed_dim[0],
                kernel_size=3,
                stride=2,
                padding=1,
                act=nn.GELU,
                bias=None,
            ),
        )

        patches_resolution = [
            img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        ]
        self.patches_resolution = patches_resolution
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):

            layer = BasicLayer(
                dim=embed_dim[i_layer],
                out_dim=embed_dim[i_layer + 1] if
                (i_layer < self.num_layers - 1) else None,
                input_resolution=patches_resolution,
                depth=depths[i_layer],
                mlp_ratio=self.mlp_ratio,
                drop=drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchEmbed if
                (i_layer < self.num_layers - 1) else None,
                downsample_kernel=sub_k[i_layer],
                focal_level=focal_levels[i_layer],
                focal_window=focal_windows[i_layer],
                use_conv_embed=use_conv_embed,
                use_checkpoint=use_checkpoint,
                use_layerscale=use_layerscale,
                layerscale_value=layerscale_value,
                use_postln=use_postln,
                use_postln_in_modulation=use_postln_in_modulation,
                normalize_modulator=normalize_modulator,
            )
            patches_resolution = [
                patches_resolution[0] // sub_k[i_layer][0],
                patches_resolution[1] // sub_k[i_layer][1]
            ]
            self.layers.append(layer)
        self.out_channels = self.num_features
        self.last_stage = last_stage
        if last_stage:
            self.out_channels = out_channels
            self.last_conv = nn.Linear(self.num_features,
                                       self.out_channels,
                                       bias=False)
            self.hardswish = nn.Hardswish()
            self.dropout = nn.Dropout(p=0.1)
            # self.avg_pool = nn.AdaptiveAvgPool2d([1, out_char_num])
            # self.last_conv = nn.Conv2d(
            #     in_channels=self.num_features,
            #     out_channels=self.out_channels,
            #     kernel_size=1,
            #     stride=1,
            #     padding=0,
            #     bias=False,
            # )
            # self.hardswish = nn.Hardswish()
            # self.dropout = nn.Dropout(p=0.1)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight,
                                    mode='fan_out',
                                    nonlinearity='relu')

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'patch_embed', 'downsample'}

    def forward(self, x):
        if len(x.shape) == 5:
            x = x.flatten(0, 1)
        x = self.patch_embed(x)
        H, W = x.shape[2:]
        x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
        x = self.pos_drop(x)

        for layer in self.layers:
            x, H, W = layer(x, H, W)

        if self.feat2d:
            x = x.transpose(1, 2).reshape(-1, self.num_features, H, W)

        if self.last_stage:

            x = x.reshape(-1, H, W, self.num_features).mean(1)
            x = self.last_conv(x)
            x = self.hardswish(x)
            x = self.dropout(x)
            # x = self.avg_pool(x.transpose(1, 2).reshape(-1, self.num_features, H, W))
            # x = self.last_conv(x)
            # x = self.hardswish(x)
            # x = self.dropout(x)
            # x = x.flatten(2).transpose(1, 2)
        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[
            0] * self.patches_resolution[1] // (2**self.num_layers)
        return flops
