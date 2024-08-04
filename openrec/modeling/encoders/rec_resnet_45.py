import math

import numpy as np
import torch.nn as nn

from openrec.modeling.common import Block


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding."""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet45(nn.Module):

    def __init__(
        self,
        in_channels=3,
        block=BasicBlock,
        layers=[3, 4, 6, 6, 3],
        strides=[2, 1, 2, 1, 1],
        last_stage=False,
        out_channels=256,
        trans_layer=0,
        out_dim=384,
        feat2d=True,
        return_list=False,
    ):
        super(ResNet45, self).__init__()
        self.inplanes = 32
        self.conv1 = nn.Conv2d(in_channels,
                               32,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 32, layers[0], stride=strides[0])
        self.layer2 = self._make_layer(block, 64, layers[1], stride=strides[1])
        self.layer3 = self._make_layer(block,
                                       128,
                                       layers[2],
                                       stride=strides[2])
        self.layer4 = self._make_layer(block,
                                       256,
                                       layers[3],
                                       stride=strides[3])
        self.layer5 = self._make_layer(block,
                                       512,
                                       layers[4],
                                       stride=strides[4])
        self.out_channels = 512
        self.feat2d = feat2d
        self.return_list = return_list
        if trans_layer > 0:
            dpr = np.linspace(0, 0.1, trans_layer)
            blocks = [nn.Linear(512, out_dim)] + [
                Block(dim=out_dim,
                      num_heads=out_dim // 32,
                      mlp_ratio=4.0,
                      qkv_bias=False,
                      drop_path=dpr[i]) for i in range(trans_layer)
            ]
            self.trans_blocks = nn.Sequential(*blocks)
            dim = out_dim
            self.out_channels = out_dim
        else:
            self.trans_blocks = None
            dim = 512
        self.last_stage = last_stage
        if last_stage:
            self.out_channels = out_channels
            self.last_conv = nn.Linear(dim, self.out_channels, bias=False)
            self.hardswish = nn.Hardswish()
            self.dropout = nn.Dropout(p=0.1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, mean=0, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x2 = self.layer2(x)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)

        if self.return_list:
            return [x2, x3, x4, x5]
        x = x5
        if self.trans_blocks is not None:
            B, C, H, W = x.shape
            x = self.trans_blocks(x.flatten(2, 3).transpose(1, 2))
            x = x.transpose(1, 2).reshape(B, -1, H, W)

        if self.last_stage:
            x = x.mean(2).transpose(1, 2)
            x = self.last_conv(x)
            x = self.hardswish(x)
            x = self.dropout(x)
        elif not self.feat2d:
            x = x.flatten(2).transpose(1, 2)
        return x
