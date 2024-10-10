import torch
import torch.nn as nn
import torch.nn.functional as F

from openrec.modeling.encoders.svtrnet import (
    Block,
    ConvBNLayer,
    kaiming_normal_,
    trunc_normal_,
    zeros_,
    ones_,
)


class Swish(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * F.sigmoid(x)


class EncoderWithSVTR(nn.Module):

    def __init__(
        self,
        in_channels,
        dims=64,  # XS
        depth=2,
        hidden_dims=120,
        use_guide=False,
        num_heads=8,
        qkv_bias=True,
        mlp_ratio=2.0,
        drop_rate=0.1,
        attn_drop_rate=0.1,
        drop_path=0.0,
        kernel_size=[3, 3],
        qk_scale=None,
        use_pool=True,
    ):
        super(EncoderWithSVTR, self).__init__()
        self.depth = depth
        self.use_guide = use_guide
        self.use_pool = use_pool
        self.conv1 = ConvBNLayer(
            in_channels,
            in_channels // 8,
            kernel_size=kernel_size,
            padding=[kernel_size[0] // 2, kernel_size[1] // 2],
            act=Swish,
            bias=False)
        self.conv2 = ConvBNLayer(in_channels // 8,
                                 hidden_dims,
                                 kernel_size=1,
                                 act=Swish,
                                 bias=False)

        self.svtr_block = nn.ModuleList([
            Block(
                dim=hidden_dims,
                num_heads=num_heads,
                mixer='Global',
                HW=None,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                act_layer=Swish,
                attn_drop=attn_drop_rate,
                drop_path=drop_path,
                norm_layer='nn.LayerNorm',
                eps=1e-05,
                prenorm=False,
            ) for i in range(depth)
        ])
        self.norm = nn.LayerNorm(hidden_dims, eps=1e-6)
        self.conv3 = ConvBNLayer(hidden_dims,
                                 in_channels,
                                 kernel_size=1,
                                 act=Swish,
                                 bias=False)
        # last conv-nxn, the input is concat of input tensor and conv3 output tensor
        self.conv4 = ConvBNLayer(
            2 * in_channels,
            in_channels // 8,
            kernel_size=kernel_size,
            padding=[kernel_size[0] // 2, kernel_size[1] // 2],
            act=Swish,
            bias=False)

        self.conv1x1 = ConvBNLayer(in_channels // 8,
                                   dims,
                                   kernel_size=1,
                                   act=Swish,
                                   bias=False)
        self.out_channels = dims
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

    def pool_h_2(self, x):
        # x: B, C, H, W
        x = x.mean(dim=2, keepdim=True)
        x = F.avg_pool2d(x, kernel_size=(1, 2))
        return x  # B, C, 1, W//2

    def forward(self, x):

        if self.use_pool:
            x = self.pool_h_2(x)

        # for use guide
        if self.use_guide:
            z = x.detach()
        else:
            z = x
        # for short cut
        h = z
        # reduce dim
        z = self.conv1(z)
        z = self.conv2(z)
        # SVTR global block
        B, C, H, W = z.shape
        z = z.flatten(2).transpose(1, 2)
        for blk in self.svtr_block:
            z = blk(z)
        z = self.norm(z)
        # last stage
        z = z.reshape(-1, H, W, C).permute(0, 3, 1, 2)
        z = self.conv3(z)
        z = torch.concat((h, z), dim=1)
        z = self.conv1x1(self.conv4(z))
        return z


class CTCDecoder(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels=6625,
                 mid_channels=None,
                 return_feats=False,
                 svtr_encoder=None,
                 **kwargs):
        super(CTCDecoder, self).__init__()
        if svtr_encoder is not None:
            svtr_encoder['in_channels'] = in_channels
            self.svtr_encoder = EncoderWithSVTR(**svtr_encoder)
            in_channels = self.svtr_encoder.out_channels
        else:
            self.svtr_encoder = None
        if mid_channels is None:
            self.fc = nn.Linear(
                in_channels,
                out_channels,
            )
        else:
            self.fc1 = nn.Linear(
                in_channels,
                mid_channels,
            )
            self.fc2 = nn.Linear(
                mid_channels,
                out_channels,
            )

        self.out_channels = out_channels
        self.mid_channels = mid_channels
        self.return_feats = return_feats

    def forward(self, x, data=None):

        if self.svtr_encoder is not None:
            x = self.svtr_encoder(x)
            x = x.flatten(2).transpose(1, 2)

        if self.mid_channels is None:
            predicts = self.fc(x)
        else:
            x = self.fc1(x)
            predicts = self.fc2(x)

        if self.return_feats:
            result = (x, predicts)
        else:
            result = predicts

        if not self.training:
            predicts = F.softmax(predicts, dim=2)
            result = predicts

        return result
