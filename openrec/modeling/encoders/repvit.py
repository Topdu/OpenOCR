"""
This code is refer from:
https://github.com/THU-MIG/RepViT
"""

import torch.nn as nn
import torch
from torch.nn.init import constant_


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def make_divisible(v, divisor=8, min_value=None, round_limit=0.9):
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < round_limit * v:
        new_v += divisor
    return new_v


class SEModule(nn.Module):
    """SE Module as defined in original SE-Nets with a few additions
    Additions include:
        * divisor can be specified to keep channels % div == 0 (default: 8)
        * reduction channels can be specified directly by arg (if rd_channels is set)
        * reduction channels can be specified by float rd_ratio (default: 1/16)
        * global max pooling can be added to the squeeze aggregation
        * customizable activation, normalization, and gate layer
    """

    def __init__(
        self,
        channels,
        rd_ratio=1.0 / 16,
        rd_channels=None,
        rd_divisor=8,
        act_layer=nn.ReLU,
    ):
        super(SEModule, self).__init__()
        if not rd_channels:
            rd_channels = make_divisible(channels * rd_ratio,
                                         rd_divisor,
                                         round_limit=0.0)
        self.fc1 = nn.Conv2d(channels, rd_channels, kernel_size=1, bias=True)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(rd_channels, channels, kernel_size=1, bias=True)

    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        x_se = self.fc1(x_se)
        x_se = self.act(x_se)
        x_se = self.fc2(x_se)
        return x * torch.sigmoid(x_se)


class Conv2D_BN(nn.Sequential):

    def __init__(
        self,
        a,
        b,
        ks=1,
        stride=1,
        pad=0,
        dilation=1,
        groups=1,
        bn_weight_init=1,
        resolution=-10000,
    ):
        super().__init__()
        self.add_module(
            'c', nn.Conv2d(a, b, ks, stride, pad, dilation, groups,
                           bias=False))
        self.add_module('bn', nn.BatchNorm2d(b))
        constant_(self.bn.weight, bn_weight_init)
        constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps)**0.5
        m = nn.Conv2d(w.size(1) * self.c.groups,
                      w.size(0),
                      w.shape[2:],
                      stride=self.c.stride,
                      padding=self.c.padding,
                      dilation=self.c.dilation,
                      groups=self.c.groups,
                      device=c.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class Residual(torch.nn.Module):

    def __init__(self, m, drop=0.):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:
            return x + self.m(x) * torch.rand(
                x.size(0), 1, 1, 1, device=x.device).ge_(
                    self.drop).div(1 - self.drop).detach()
        else:
            return x + self.m(x)

    @torch.no_grad()
    def fuse(self):
        if isinstance(self.m, Conv2D_BN):
            m = self.m.fuse()
            assert (m.groups == m.in_channels)
            identity = torch.ones(m.weight.shape[0], m.weight.shape[1], 1, 1)
            identity = nn.functional.pad(identity, [1, 1, 1, 1])
            m.weight += identity.to(m.weight.device)
            return m
        elif isinstance(self.m, nn.Conv2d):
            m = self.m
            assert (m.groups != m.in_channels)
            identity = torch.ones(m.weight.shape[0], m.weight.shape[1], 1, 1)
            identity = nn.functional.pad(identity, [1, 1, 1, 1])
            m.weight += identity.to(m.weight.device)
            return m
        else:
            return self


class RepVGGDW(nn.Module):

    def __init__(self, ed) -> None:
        super().__init__()
        self.conv = Conv2D_BN(ed, ed, 3, 1, 1, groups=ed)
        self.conv1 = nn.Conv2d(ed, ed, 1, 1, 0, groups=ed)
        self.dim = ed
        self.bn = nn.BatchNorm2d(ed)

    def forward(self, x):
        return self.bn((self.conv(x) + self.conv1(x)) + x)

    @torch.no_grad()
    def fuse(self):
        conv = self.conv.fuse()
        conv1 = self.conv1

        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias

        conv1_w = nn.functional.pad(conv1_w, [1, 1, 1, 1])

        identity = nn.functional.pad(
            torch.ones(conv1_w.shape[0],
                       conv1_w.shape[1],
                       1,
                       1,
                       device=conv1_w.device), [1, 1, 1, 1])

        final_conv_w = conv_w + conv1_w + identity
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)

        bn = self.bn
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = conv.weight * w[:, None, None, None]
        b = bn.bias + (conv.bias - bn.running_mean) * bn.weight / \
            (bn.running_var + bn.eps)**0.5
        conv.weight.data.copy_(w)
        conv.bias.data.copy_(b)
        return conv


class RepViTBlock(nn.Module):

    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se,
                 use_hs):
        super(RepViTBlock, self).__init__()

        self.identity = stride == 1 and inp == oup
        assert hidden_dim == 2 * inp

        if stride != 1:
            self.token_mixer = nn.Sequential(
                Conv2D_BN(inp,
                          inp,
                          kernel_size,
                          stride, (kernel_size - 1) // 2,
                          groups=inp),
                SEModule(inp, 0.25) if use_se else nn.Identity(),
                Conv2D_BN(inp, oup, ks=1, stride=1, pad=0),
            )
            self.channel_mixer = Residual(
                nn.Sequential(
                    # pw
                    Conv2D_BN(oup, 2 * oup, 1, 1, 0),
                    nn.GELU() if use_hs else nn.GELU(),
                    # pw-linear
                    Conv2D_BN(2 * oup, oup, 1, 1, 0, bn_weight_init=0),
                ))
        else:
            assert self.identity
            self.token_mixer = nn.Sequential(
                RepVGGDW(inp),
                SEModule(inp, 0.25) if use_se else nn.Identity(),
            )
            self.channel_mixer = Residual(
                nn.Sequential(
                    # pw
                    Conv2D_BN(inp, hidden_dim, 1, 1, 0),
                    nn.GELU() if use_hs else nn.GELU(),
                    # pw-linear
                    Conv2D_BN(hidden_dim, oup, 1, 1, 0, bn_weight_init=0),
                ))

    def forward(self, x):
        return self.channel_mixer(self.token_mixer(x))


class RepViT(nn.Module):

    def __init__(self, cfgs, in_channels=3, out_indices=None):
        super(RepViT, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = cfgs

        # building first layer
        input_channel = self.cfgs[0][2]
        patch_embed = nn.Sequential(
            Conv2D_BN(in_channels, input_channel // 2, 3, 2, 1),
            nn.GELU(),
            Conv2D_BN(input_channel // 2, input_channel, 3, 2, 1),
        )
        layers = [patch_embed]
        # building inverted residual blocks
        block = RepViTBlock
        for k, t, c, use_se, use_hs, s in self.cfgs:
            output_channel = _make_divisible(c, 8)
            exp_size = _make_divisible(input_channel * t, 8)
            layers.append(
                block(input_channel, exp_size, output_channel, k, s, use_se,
                      use_hs))
            input_channel = output_channel
        self.features = nn.ModuleList(layers)
        self.out_indices = out_indices
        if out_indices is not None:
            self.out_channels = [self.cfgs[ids - 1][2] for ids in out_indices]
        else:
            self.out_channels = self.cfgs[-1][2]

    def forward(self, x):
        if self.out_indices is not None:
            return self.forward_det(x)
        return self.forward_rec(x)

    def forward_det(self, x):
        outs = []
        for i, f in enumerate(self.features):
            x = f(x)
            if i in self.out_indices:
                outs.append(x)
        return outs

    def forward_rec(self, x):
        for f in self.features:
            x = f(x)
        return x


def RepSVTREncoder(in_channels=3):
    """
    Constructs a MobileNetV3-Large model
    """
    # k, t, c, SE, HS, s
    cfgs = [
        [3, 2, 96, 1, 0, 1],
        [3, 2, 96, 0, 0, 1],
        [3, 2, 96, 0, 0, 1],
        [3, 2, 192, 0, 1, (2, 1)],
        [3, 2, 192, 1, 1, 1],
        [3, 2, 192, 0, 1, 1],
        [3, 2, 192, 1, 1, 1],
        [3, 2, 192, 0, 1, 1],
        [3, 2, 192, 1, 1, 1],
        [3, 2, 192, 0, 1, 1],
        [3, 2, 384, 0, 1, (2, 1)],
        [3, 2, 384, 1, 1, 1],
        [3, 2, 384, 0, 1, 1],
    ]
    return RepViT(cfgs, in_channels=in_channels)


def RepSVTR_det(in_channels=3, out_indices=[2, 5, 10, 13]):
    """
    Constructs a MobileNetV3-Large model
    """
    # k, t, c, SE, HS, s
    cfgs = [
        [3, 2, 48, 1, 0, 1],
        [3, 2, 48, 0, 0, 1],
        [3, 2, 96, 0, 0, 2],
        [3, 2, 96, 1, 0, 1],
        [3, 2, 96, 0, 0, 1],
        [3, 2, 192, 0, 1, 2],
        [3, 2, 192, 1, 1, 1],
        [3, 2, 192, 0, 1, 1],
        [3, 2, 192, 1, 1, 1],
        [3, 2, 192, 0, 1, 1],
        [3, 2, 384, 0, 1, 2],
        [3, 2, 384, 1, 1, 1],
        [3, 2, 384, 0, 1, 1],
    ]
    return RepViT(cfgs, in_channels=in_channels, out_indices=out_indices)
