"""This code is refer from:
https://github.com/MelosY/CAM
"""

# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_

from openrec.modeling.common import DropPath


class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self,
                 normalized_shape,
                 eps=1e-6,
                 data_format='channels_last'):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ['channels_last', 'channels_first']:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == 'channels_last':
            return F.layer_norm(x, self.normalized_shape, self.weight,
                                self.bias, self.eps)
        elif self.data_format == 'channels_first':
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """

    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, inputs, mask=None):
        x = inputs
        if mask is not None:
            x = x * (1. - mask)
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (inputs * Nx) + self.beta + inputs


class Block(nn.Module):
    """ ConvNeXtV2 Block.

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """

    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3,
                                groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(
            dim,
            4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x.contiguous())
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class ConvNeXtV2(nn.Module):
    """ ConvNeXt V2

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.

    """

    def __init__(
        self,
        in_channels=3,
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768],
        drop_path_rate=0.,
        strides=[(4, 4), (2, 2), (2, 2), (2, 2)],
    ):
        super().__init__()
        self.strides = strides
        self.depths = depths
        self.downsample_layers = nn.ModuleList(
        )  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_channels,
                      dims[0],
                      kernel_size=strides[0],
                      stride=strides[0]),
            LayerNorm(dims[0], eps=1e-6, data_format='channels_first'))
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format='channels_first'),
                nn.Conv2d(dims[i],
                          dims[i + 1],
                          kernel_size=strides[i + 1],
                          stride=strides[i + 1]),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList(
        )  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(*[
                Block(dim=dims[i], drop_path=dp_rates[cur + j])
                for j in range(depths[i])
            ])
            self.stages.append(stage)
            cur += depths[i]

        self.out_channels = dims[-1]

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, (nn.Conv2d, nn.Linear)) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        if isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight,
                                    mode='fan_out',
                                    nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.SyncBatchNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def no_weight_decay(self):
        return {}

    def forward(self, x):
        feats = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            feats.append(x)
        return feats


class Swish(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class UNetBlock(nn.Module):

    def __init__(self, cin, cout, bn2d, stride, deformable=False):
        """
        a UNet block with 2x up sampling
        """
        super().__init__()
        stride_h, stride_w = stride
        if stride_h == 1:
            kernel_h = 1
            padding_h = 0
        elif stride_h == 2:
            kernel_h = 4
            padding_h = 1
        elif stride_h == 4:
            kernel_h = 4
            padding_h = 0

        if stride_w == 1:
            kernel_w = 1
            padding_w = 0
        elif stride_w == 2:
            kernel_w = 4
            padding_w = 1
        elif stride_w == 4:
            kernel_w = 4
            padding_w = 0

        conv = nn.Conv2d

        self.up_sample = nn.ConvTranspose2d(cin,
                                            cin,
                                            kernel_size=(kernel_h, kernel_w),
                                            stride=(stride_h, stride_w),
                                            padding=(padding_h, padding_w),
                                            bias=True)
        self.conv = nn.Sequential(
            conv(cin, cin, kernel_size=3, stride=1, padding=1, bias=False),
            bn2d(cin),
            nn.ReLU6(inplace=True),
            conv(cin, cout, kernel_size=3, stride=1, padding=1, bias=False),
            bn2d(cout),
        )

    def forward(self, x):
        x = self.up_sample(x)
        return self.conv(x)


class DepthWiseUNetBlock(nn.Module):

    def __init__(self, cin, cout, bn2d, stride, deformable=False):
        """
        a UNet block with 2x up sampling
        """
        super().__init__()
        stride_h, stride_w = stride
        if stride_h == 1:
            kernel_h = 1
            padding_h = 0
        elif stride_h == 2:
            kernel_h = 4
            padding_h = 1
        elif stride_h == 4:
            kernel_h = 4
            padding_h = 0

        if stride_w == 1:
            kernel_w = 1
            padding_w = 0
        elif stride_w == 2:
            kernel_w = 4
            padding_w = 1
        elif stride_w == 4:
            kernel_w = 4
            padding_w = 0

        self.up_sample = nn.ConvTranspose2d(cin,
                                            cin,
                                            kernel_size=(kernel_h, kernel_w),
                                            stride=(stride_h, stride_w),
                                            padding=(padding_h, padding_w),
                                            bias=True)
        self.conv = nn.Sequential(
            nn.Conv2d(cin,
                      cin,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False,
                      groups=cin),
            nn.Conv2d(cin, cin, kernel_size=1, stride=1, padding=0,
                      bias=False),
            bn2d(cin),
            nn.ReLU6(inplace=True),
            nn.Conv2d(cin,
                      cin,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False,
                      groups=cin),
            nn.Conv2d(cin,
                      cout,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=False),
            bn2d(cout),
        )

    def forward(self, x):
        x = self.up_sample(x)
        return self.conv(x)


class SFTLayer(nn.Module):

    def __init__(self, dim_in, dim_out):
        super(SFTLayer, self).__init__()
        self.SFT_scale_conv0 = nn.Linear(
            dim_in,
            dim_in,
        )
        self.SFT_scale_conv1 = nn.Linear(
            dim_in,
            dim_out,
        )
        self.SFT_shift_conv0 = nn.Linear(
            dim_in,
            dim_in,
        )
        self.SFT_shift_conv1 = nn.Linear(
            dim_in,
            dim_out,
        )

    def forward(self, x):
        # x[0]: fea; x[1]: cond
        scale = self.SFT_scale_conv1(
            F.leaky_relu(self.SFT_scale_conv0(x[1]), 0.1, inplace=True))
        shift = self.SFT_shift_conv1(
            F.leaky_relu(self.SFT_shift_conv0(x[1]), 0.1, inplace=True))
        return x[0] * (scale + 1) + shift


class MoreUNetBlock(nn.Module):

    def __init__(self, cin, cout, bn2d, stride, deformable=False):
        """
        a UNet block with 2x up sampling
        """
        super().__init__()
        stride_h, stride_w = stride
        if stride_h == 1:
            kernel_h = 1
            padding_h = 0
        elif stride_h == 2:
            kernel_h = 4
            padding_h = 1
        elif stride_h == 4:
            kernel_h = 4
            padding_h = 0

        if stride_w == 1:
            kernel_w = 1
            padding_w = 0
        elif stride_w == 2:
            kernel_w = 4
            padding_w = 1
        elif stride_w == 4:
            kernel_w = 4
            padding_w = 0

        self.up_sample = nn.ConvTranspose2d(cin,
                                            cin,
                                            kernel_size=(kernel_h, kernel_w),
                                            stride=(stride_h, stride_w),
                                            padding=(padding_h, padding_w),
                                            bias=True)
        self.conv = nn.Sequential(
            nn.Conv2d(cin,
                      cin,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False,
                      groups=cin),
            nn.Conv2d(cin, cin, kernel_size=1, stride=1, padding=0,
                      bias=False), bn2d(cin), nn.ReLU6(inplace=True),
            nn.Conv2d(cin,
                      cin,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False,
                      groups=cin),
            nn.Conv2d(cin, cin, kernel_size=1, stride=1, padding=0,
                      bias=False), bn2d(cin), nn.ReLU6(inplace=True),
            nn.Conv2d(cin,
                      cin,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False,
                      groups=cin),
            nn.Conv2d(cin,
                      cout,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=False), bn2d(cout), nn.ReLU6(inplace=True),
            nn.Conv2d(cout,
                      cout,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False,
                      groups=cout),
            nn.Conv2d(cout,
                      cout,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=False), bn2d(cout))

    def forward(self, x):
        x = self.up_sample(x)
        return self.conv(x)


class BinaryDecoder(nn.Module):

    def __init__(self,
                 dims,
                 num_classes,
                 strides,
                 use_depthwise_unet=False,
                 use_more_unet=False,
                 binary_loss_type='DiceLoss') -> None:
        super().__init__()

        channels = [dims[-1] // 2**i for i in range(4)]
        self.linear_enc2binary = nn.Sequential(
            nn.Conv2d(dims[-1], dims[-1], kernel_size=3, stride=1, padding=1),
            nn.SyncBatchNorm(dims[-1]),
        )
        self.strides = strides
        self.use_deformable = False
        self.binary_decoder = nn.ModuleList()
        unet = DepthWiseUNetBlock if use_depthwise_unet else UNetBlock
        unet = MoreUNetBlock if use_more_unet else unet

        for i in range(3):
            up_sample_stride = self.strides[::-1][i]
            cin, cout = channels[i], channels[i + 1]
            self.binary_decoder.append(
                unet(cin, cout, nn.SyncBatchNorm, up_sample_stride,
                     self.use_deformable))

        last_stride = (self.strides[0][0] // 2, self.strides[0][1] // 2)
        self.binary_decoder.append(
            unet(cout, cout, nn.SyncBatchNorm, last_stride,
                 self.use_deformable))

        if binary_loss_type == 'CrossEntropyDiceLoss' or binary_loss_type == 'BanlanceMultiClassCrossEntropyLoss':
            segm_num_cls = num_classes - 2
        else:
            segm_num_cls = num_classes - 3
        self.binary_pred = nn.Conv2d(channels[-1],
                                     segm_num_cls,
                                     kernel_size=1,
                                     stride=1,
                                     bias=True)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p_h, p_w = self.strides[0]
        p_h = p_h // 2
        p_w = p_w // 2
        h = imgs.shape[2] // p_h
        w = imgs.shape[3] // p_w

        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p_h, w, p_w))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p_h * p_w * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, patch_size**2, h, w)
        imgs: (N, 3, H, W)
        """
        p_h, p_w = self.strides[0]
        p_h = p_h // 2
        p_w = p_w // 2
        _, _, h, w = x.shape
        assert p_h * p_w == x.shape[1]

        x = x.permute(0, 2, 3, 1)  # [N, h, w, 4*4]
        x = x.reshape(shape=(x.shape[0], h, w, p_h, p_w))
        x = torch.einsum('nhwpq->nhpwq', x)
        imgs = x.reshape(shape=(x.shape[0], h * p_h, w * p_w))
        return imgs

    def forward(self, x, time=None):
        """
          x: the encoder feat to init the query for binary prediction, usually this is equal to the `img`.
          img: the encoder feat.
          txt: the unnormmed text to get the length of predicted words.
          txt_feat: the text feat before character prediction.
          xs: the encoder feat from different stages
        """

        binary_feats = []
        x = self.linear_enc2binary(x)
        binary_feats.append(x.clone())

        for i, d in enumerate(self.binary_decoder):

            x = d(x)
            binary_feats.append(x.clone())
        #return None,binary_feats
        x = self.binary_pred(x)

        if self.training:
            return x, binary_feats
        else:
            # return torch.sigmoid(x), binary_feat
            return x.softmax(1), binary_feats


class LayerNormProxy(nn.Module):

    def __init__(self, dim):

        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2)


class DAttentionFuse(nn.Module):

    def __init__(
        self,
        q_size=(4, 32),
        kv_size=(4, 32),
        n_heads=8,
        n_head_channels=80,
        n_groups=4,
        attn_drop=0.0,
        proj_drop=0.0,
        stride=2,
        offset_range_factor=2,
        use_pe=True,
        stage_idx=0,
    ):
        '''
        stage_idx from 2 to 3
        '''

        super().__init__()
        self.n_head_channels = n_head_channels
        self.scale = self.n_head_channels**-0.5
        self.n_heads = n_heads
        self.q_h, self.q_w = q_size
        self.kv_h, self.kv_w = kv_size
        self.nc = n_head_channels * n_heads
        self.n_groups = n_groups
        self.n_group_channels = self.nc // self.n_groups
        self.n_group_heads = self.n_heads // self.n_groups
        self.use_pe = use_pe
        self.offset_range_factor = offset_range_factor
        ksizes = [9, 7, 5, 3]
        kk = ksizes[stage_idx]

        self.conv_offset = nn.Sequential(
            nn.Conv2d(2 * self.n_group_channels,
                      2 * self.n_group_channels,
                      kk,
                      stride,
                      kk // 2,
                      groups=self.n_group_channels),
            LayerNormProxy(2 * self.n_group_channels), nn.GELU(),
            nn.Conv2d(2 * self.n_group_channels, 2, 1, 1, 0, bias=False))

        self.proj_q = nn.Conv2d(self.nc,
                                self.nc,
                                kernel_size=1,
                                stride=1,
                                padding=0)

        self.proj_k = nn.Conv2d(self.nc,
                                self.nc,
                                kernel_size=1,
                                stride=1,
                                padding=0)

        self.proj_v = nn.Conv2d(self.nc,
                                self.nc,
                                kernel_size=1,
                                stride=1,
                                padding=0)

        self.proj_out = nn.Conv2d(self.nc,
                                  self.nc,
                                  kernel_size=1,
                                  stride=1,
                                  padding=0)

        self.proj_drop = nn.Dropout(proj_drop, inplace=True)
        self.attn_drop = nn.Dropout(attn_drop, inplace=True)

        if self.use_pe:
            self.rpe_table = nn.Parameter(
                torch.zeros(self.n_heads, self.kv_h * 2 - 1,
                            self.kv_w * 2 - 1))
            trunc_normal_(self.rpe_table, std=0.01)
        else:
            self.rpe_table = None

    @torch.no_grad()
    def _get_ref_points(self, H_key, W_key, B, dtype, device):

        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_key - 0.5, H_key, dtype=dtype,
                           device=device),
            torch.linspace(0.5, W_key - 0.5, W_key, dtype=dtype,
                           device=device))
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W_key).mul_(2).sub_(1)
        ref[..., 0].div_(H_key).mul_(2).sub_(1)
        ref = ref[None, ...].expand(B * self.n_groups, -1, -1,
                                    -1)  # B * g H W 2
        return ref

    def forward(self, x, y):
        B, C, H, W = x.size()
        dtype, device = x.dtype, x.device

        q_off = torch.cat(
            (x, y), dim=1
        ).reshape(B, self.n_groups, 2 * self.n_group_channels, H, W).flatten(
            0, 1
        )  #einops.rearrange(torch.cat((x,y),dim=1), 'b (g c) h w -> (b g) c h w', g=self.n_groups, c=2*self.n_group_channels)

        offset = self.conv_offset(q_off)  # B * g 2 Hg Wg
        Hk, Wk = offset.size(2), offset.size(3)
        n_sample = Hk * Wk
        if self.offset_range_factor > 0:
            offset_range = torch.tensor([1.0 / Hk, 1.0 / Wk],
                                        device=device).reshape(1, 2, 1, 1)
            offset = offset.tanh().mul(offset_range).mul(
                self.offset_range_factor)

        offset = offset.permute(
            0, 2, 3, 1)  #einops.rearrange(offset, 'b p h w -> b h w p')
        reference = self._get_ref_points(Hk, Wk, B, dtype, device)

        if self.offset_range_factor >= 0:
            pos = offset + reference
        else:
            pos = (offset + reference).tanh()

        q = self.proj_q(y)
        x_sampled = F.grid_sample(
            input=x.reshape(B * self.n_groups, self.n_group_channels, H, W),
            grid=pos[..., (1, 0)],  # y, x -> x, y
            mode='bilinear',
            align_corners=False)  # B * g, Cg, Hg, Wg

        x_sampled = x_sampled.reshape(B, C, 1, n_sample)

        q = q.reshape(B * self.n_heads, self.n_head_channels, H * W)
        k = self.proj_k(x_sampled).reshape(B * self.n_heads,
                                           self.n_head_channels, n_sample)
        v = self.proj_v(x_sampled).reshape(B * self.n_heads,
                                           self.n_head_channels, n_sample)

        attn = torch.einsum('b c m, b c n -> b m n', q, k)  # B * h, HW, Ns
        attn = attn.mul(self.scale)

        if self.use_pe:
            rpe_table = self.rpe_table
            rpe_bias = rpe_table[None, ...].expand(B, -1, -1, -1)

            q_grid = self._get_ref_points(H, W, B, dtype, device)

            displacement = (
                q_grid.reshape(B * self.n_groups, H * W, 2).unsqueeze(2) -
                pos.reshape(B * self.n_groups, n_sample,
                            2).unsqueeze(1)).mul(0.5)

            attn_bias = F.grid_sample(input=rpe_bias.reshape(
                B * self.n_groups, self.n_group_heads, 2 * H - 1, 2 * W - 1),
                                      grid=displacement[..., (1, 0)],
                                      mode='bilinear',
                                      align_corners=False)

            attn_bias = attn_bias.reshape(B * self.n_heads, H * W, n_sample)

            attn = attn + attn_bias

        attn = F.softmax(attn, dim=2)
        attn = self.attn_drop(attn)

        out = torch.einsum('b m n, b c n -> b c m', attn, v)
        out = out.reshape(B, C, H, W)
        out = self.proj_drop(self.proj_out(out))

        return out, pos.reshape(B, self.n_groups, Hk, Wk,
                                2), reference.reshape(B, self.n_groups, Hk, Wk,
                                                      2)


class FuseModel(nn.Module):

    def __init__(self, dims, deform_stride=2, stage_idx=2) -> None:
        super().__init__()

        channels = [dims[-1] // 2**i for i in range(4)]

        refine_conv = nn.Conv2d
        self.deform_stride = deform_stride

        k_size = [(2, 2), (2, 1), (2, 1), (1, 1)]
        q_size = (2, 32)
        in_out_ch = [(-1, -2), (-2, -3), (-3, -4), (-4, -4)]

        self.binary_condition_layer = DAttentionFuse(
            q_size=q_size,
            kv_size=q_size,
            stride=self.deform_stride,
            n_head_channels=dims[-1] // 8,
            stage_idx=stage_idx)

        self.binary2refine_linear_norm = nn.ModuleList()
        for i in range(len(k_size)):
            self.binary2refine_linear_norm.append(
                nn.Sequential(
                    Block(dim=channels[in_out_ch[i][0]]),
                    LayerNorm(channels[in_out_ch[i][0]],
                              eps=1e-6,
                              data_format='channels_first'),
                    refine_conv(channels[in_out_ch[i][0]],
                                channels[in_out_ch[i][1]],
                                kernel_size=k_size[i],
                                stride=k_size[i])),  # [8, 32]
            )

    def forward(self, recog_feat, binary_feats, dec_in=None):
        multi_feat = []
        binary_feat = binary_feats[-1]
        for i in range(len(self.binary2refine_linear_norm)):
            binary_feat = self.binary2refine_linear_norm[i](binary_feat)
            multi_feat.append(binary_feat)
        binary_feat = binary_feat + binary_feats[0]
        multi_feat[3] += binary_feats[0]
        binary_refined_feat, pos, _ = self.binary_condition_layer(
            recog_feat, binary_feat)
        binary_refined_feat = binary_refined_feat + binary_feat
        return binary_refined_feat, binary_feat


# def convnextv2_nano(**kwargs):
#     model = ConvNeXtV2(depths=[2, 2, 8, 2], dims=[80, 160, 320, 640], **kwargs)
#     return model


class CAMEncoder(nn.Module):
    """

    Args:
        in_chans (int): Number of input image channels. Default: 3
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.

    """

    def __init__(self,
                 in_channels=3,
                 depths=[2, 2, 8, 2],
                 dims=[80, 160, 320, 640],
                 nb_classes=71,
                 drop_path_rate=0.2,
                 strides=[(4, 4), (2, 1), (2, 1), (1, 1)],
                 deform_stride=2,
                 stage_idx=2,
                 use_depthwise_unet=True,
                 use_more_unet=False,
                 binary_loss_type='BanlanceMultiClassCrossEntropyLoss',
                 mid_size=True,
                 d_embedding=384):
        super().__init__()

        self.backbone = ConvNeXtV2(in_channels=in_channels,
                                   depths=depths,
                                   dims=dims,
                                   drop_path_rate=drop_path_rate,
                                   strides=strides)

        self.mid_size = mid_size
        if self.mid_size:
            self.enc_downsample = nn.Sequential(
                nn.Conv2d(dims[-1], dims[-1] // 2, kernel_size=1, stride=1),
                nn.SyncBatchNorm(dims[-1] // 2),
                #nn.ReLU6(inplace=True),
                nn.Conv2d(dims[-1] // 2,
                          dims[-1] // 2,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          bias=False,
                          groups=dims[-1] // 2),
                nn.Conv2d(dims[-1] // 2,
                          dims[-1] // 2,
                          kernel_size=1,
                          stride=1,
                          padding=0,
                          bias=False),
                nn.SyncBatchNorm(dims[-1] // 2),
            )
            dims[-1] = dims[-1] // 2
            # recognition decoder
            self.linear_enc2recog = nn.Sequential(
                nn.Conv2d(
                    dims[-1],
                    dims[-1],
                    kernel_size=1,
                    stride=1,
                ),
                nn.SyncBatchNorm(dims[-1]),
                #nn.ReLU6(inplace=True),
                nn.Conv2d(dims[-1],
                          dims[-1],
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          bias=False,
                          groups=dims[-1]),
                nn.Conv2d(dims[-1],
                          dims[-1],
                          kernel_size=1,
                          stride=1,
                          padding=0,
                          bias=False),
                nn.SyncBatchNorm(dims[-1]),
            )
        else:
            self.linear_enc2recog = nn.Sequential(
                nn.Conv2d(dims[-1], dims[-1] // 2, kernel_size=1, stride=1),
                nn.SyncBatchNorm(dims[-1] // 2),
                #nn.ReLU6(inplace=True),
                nn.Conv2d(dims[-1] // 2,
                          dims[-1],
                          kernel_size=3,
                          stride=1,
                          padding=1),
                nn.SyncBatchNorm(dims[-1]),
            )

        self.linear_norm = nn.Sequential(
            nn.Linear(dims[-1], d_embedding),
            nn.LayerNorm(d_embedding, eps=1e-6),
        )
        self.out_channels = d_embedding

        self.binary_decoder = BinaryDecoder(
            dims,
            nb_classes,
            strides,
            use_depthwise_unet=use_depthwise_unet,
            use_more_unet=use_more_unet,
            binary_loss_type=binary_loss_type)
        self.fuse_model = FuseModel(dims,
                                    deform_stride=deform_stride,
                                    stage_idx=stage_idx)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, (nn.Conv2d, nn.Linear)) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        if isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight,
                                    mode='fan_out',
                                    nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.SyncBatchNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def no_weight_decay(self):
        return {}

    def forward(self, x):
        output = {}
        enc_feats = self.backbone(x)
        enc_feat = enc_feats[-1]
        if self.mid_size:
            enc_feat = self.enc_downsample(enc_feat)
        output['enc_feat'] = enc_feat

        # binary mask
        pred_binary, binary_feats = self.binary_decoder(enc_feat)
        output['pred_binary'] = pred_binary

        reg_feat = self.linear_enc2recog(enc_feat)
        B, C, H, W = reg_feat.shape
        last_feat, binary_feat = self.fuse_model(reg_feat, binary_feats)

        dec_in = last_feat.reshape(B, C, H * W).permute(0, 2, 1)
        dec_in = self.linear_norm(dec_in)

        output['refined_feat'] = dec_in
        output['binary_feat'] = binary_feats[-1]
        return output
