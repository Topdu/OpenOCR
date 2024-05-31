import numpy as np
import torch
import torch.nn as nn


class ConvBNLayer(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel,
                 stride=1,
                 act='ReLU'):
        super(ConvBNLayer, self).__init__()
        self.act_flag = act
        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size=2 if stride == (1, 1) else kernel,
                              stride=stride,
                              padding=(kernel - 1) // 2,
                              dilation=2 if stride == (1, 1) else 1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.act_flag != 'None':
            x = self.act(x)
        return x


class Shortcut(nn.Module):

    def __init__(self, in_channels, out_channels, stride, is_first=False):
        super(Shortcut, self).__init__()
        self.use_conv = True
        if in_channels != out_channels or stride != 1 or is_first is True:
            if stride == (1, 1):
                self.conv = ConvBNLayer(in_channels, out_channels, 1, 1)
            else:
                self.conv = ConvBNLayer(in_channels, out_channels, 1, stride)
        else:
            self.use_conv = False

    def forward(self, x):
        if self.use_conv:
            x = self.conv(x)
        return x


class BottleneckBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride):
        super(BottleneckBlock, self).__init__()
        self.conv0 = ConvBNLayer(in_channels, out_channels, kernel=1)
        self.conv1 = ConvBNLayer(out_channels,
                                 out_channels,
                                 kernel=3,
                                 stride=stride)
        self.conv2 = ConvBNLayer(out_channels,
                                 out_channels * 4,
                                 kernel=1,
                                 act='None')
        self.short = Shortcut(in_channels, out_channels * 4, stride=stride)
        self.out_channels = out_channels * 4
        self.relu = nn.ReLU(True)

    def forward(self, x):
        y = self.conv0(x)
        y = self.conv1(y)
        y = self.conv2(y)
        y = y + self.short(x)
        y = self.relu(y)
        return y


class BasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride, is_first):
        super(BasicBlock, self).__init__()
        self.conv0 = ConvBNLayer(in_channels,
                                 out_channels,
                                 kernel=3,
                                 stride=stride)
        self.conv1 = ConvBNLayer(out_channels,
                                 out_channels,
                                 kernel=3,
                                 act='None')
        self.short = Shortcut(in_channels, out_channels, stride, is_first)
        self.out_chanels = out_channels
        self.relu = nn.ReLU(True)

    def forward(self, x):
        y = self.conv0(x)
        y = self.conv1(y)
        y = y + self.short(x)
        y = self.relu(y)
        return y


class ResNet_FPN(nn.Module):

    def __init__(self, in_channels=1, layers=50, **kwargs):
        super(ResNet_FPN, self).__init__()
        supported_layers = {
            18: {
                'depth': [2, 2, 2, 2],
                'block_class': BasicBlock
            },
            34: {
                'depth': [3, 4, 6, 3],
                'block_class': BasicBlock
            },
            50: {
                'depth': [3, 4, 6, 3],
                'block_class': BottleneckBlock
            },
            101: {
                'depth': [3, 4, 23, 3],
                'block_class': BottleneckBlock
            },
            152: {
                'depth': [3, 8, 36, 3],
                'block_class': BottleneckBlock
            }
        }
        stride_list = [(2, 2), (
            2,
            2,
        ), (1, 1), (1, 1)]
        num_filters = [64, 128, 256, 512]
        self.depth = supported_layers[layers]['depth']
        self.F = []
        # print(f"in_channels:{in_channels}")
        self.conv = ConvBNLayer(in_channels=in_channels,
                                out_channels=64,
                                kernel=7,
                                stride=2)  #64*256 ->32*128

        self.block_list = nn.ModuleList()
        in_ch = 64
        if layers >= 50:
            for block in range(len(self.depth)):
                for i in range(self.depth[block]):
                    self.block_list.append(
                        BottleneckBlock(
                            in_channels=in_ch,
                            out_channels=num_filters[block],
                            stride=stride_list[block] if i == 0 else 1))
                    in_ch = num_filters[block] * 4

        else:
            for block in range(len(self.depth)):
                for i in range(self.depth[block]):
                    if i == 0 and block != 0:
                        stride = (2, 1)
                    else:
                        stride = (1, 1)
                    basic_block = BasicBlock(
                        in_channels=in_ch,
                        out_channels=num_filters[block],
                        stride=stride_list[block] if i == 0 else 1,
                        is_first=block == i == 0)
                    in_ch = basic_block.out_chanels
                    self.block_list.append(basic_block)

        out_ch_list = [in_ch // 4, in_ch // 2, in_ch]
        self.base_block = nn.ModuleList()
        self.conv_trans = []
        self.bn_block = []
        for i in [-2, -3]:
            in_channels = out_ch_list[i + 1] + out_ch_list[i]
            self.base_block.append(
                nn.Conv2d(in_channels, out_ch_list[i], kernel_size=1))  #进行升通道
            self.base_block.append(
                nn.Conv2d(out_ch_list[i],
                          out_ch_list[i],
                          kernel_size=3,
                          padding=1))  #进行合并
            self.base_block.append(
                nn.Sequential(nn.BatchNorm2d(out_ch_list[i]), nn.ReLU(True)))
        self.base_block.append(nn.Conv2d(out_ch_list[i], 512, kernel_size=1))

        self.out_channels = 512

    def forward(self, x):

        # print(f"before resnetfpn x.shape:{x.shape}")
        x = self.conv(x)
        fpn_list = []
        F = []
        for i in range(len(self.depth)):
            fpn_list.append(np.sum(self.depth[:i + 1]))
        for i, block in enumerate(self.block_list):
            x = block(x)

            for number in fpn_list:
                if i + 1 == number:
                    F.append(x)
        base = F[-1]

        j = 0
        for i, block in enumerate(self.base_block):
            if i % 3 == 0 and i < 6:
                j = j + 1
                b, c, w, h = F[-j - 1].size()
                if [w, h] == list(base.size()[2:]):
                    base = base
                else:
                    base = self.conv_trans[j - 1](base)
                    base = self.bn_block[j - 1](base)
                base = torch.cat([base, F[-j - 1]], dim=1)
            base = block(base)

        return base
