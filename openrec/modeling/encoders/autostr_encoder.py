from collections import OrderedDict
import torch
import torch.nn as nn


class IdentityLayer(nn.Module):

    def __init__(self):
        super(IdentityLayer, self).__init__()

    def forward(self, x):
        return x

    @staticmethod
    def is_zero_layer():
        return False


class ZeroLayer(nn.Module):

    def __init__(self, stride):
        super(ZeroLayer, self).__init__()
        self.stride = stride

    def forward(self, x):
        n, c, h, w = x.shape
        h //= self.stride[0]
        w //= self.stride[1]
        device = x.device
        padding = torch.zeros(n, c, h, w, device=device, requires_grad=False)
        return padding

    @staticmethod
    def is_zero_layer():
        return True

    def get_flops(self, x):
        return 0, self.forward(x)


def get_same_padding(kernel_size):
    if isinstance(kernel_size, tuple):
        assert len(kernel_size) == 2, 'invalid kernel size: %s' % kernel_size
        p1 = get_same_padding(kernel_size[0])
        p2 = get_same_padding(kernel_size[1])
        return p1, p2
    assert isinstance(kernel_size,
                      int), 'kernel size should be either `int` or `tuple`'
    assert kernel_size % 2 > 0, 'kernel size should be odd number'
    return kernel_size // 2


class MBInvertedConvLayer(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=(1, 1),
                 expand_ratio=6,
                 mid_channels=None):
        super(MBInvertedConvLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.expand_ratio = expand_ratio
        self.mid_channels = mid_channels

        feature_dim = round(
            self.in_channels *
            self.expand_ratio) if mid_channels is None else mid_channels
        if self.expand_ratio == 1:
            self.inverted_bottleneck = None
        else:
            self.inverted_bottleneck = nn.Sequential(
                OrderedDict([
                    ('conv',
                     nn.Conv2d(self.in_channels,
                               feature_dim,
                               1,
                               1,
                               0,
                               bias=False)),
                    ('bn', nn.BatchNorm2d(feature_dim)),
                    ('act', nn.ReLU6(inplace=True)),
                ]))
        pad = get_same_padding(self.kernel_size)
        self.depth_conv = nn.Sequential(
            OrderedDict([
                ('conv',
                 nn.Conv2d(feature_dim,
                           feature_dim,
                           kernel_size,
                           stride,
                           pad,
                           groups=feature_dim,
                           bias=False)),
                ('bn', nn.BatchNorm2d(feature_dim)),
                ('act', nn.ReLU6(inplace=True)),
            ]))
        self.point_conv = nn.Sequential(
            OrderedDict([
                ('conv',
                 nn.Conv2d(feature_dim, out_channels, 1, 1, 0, bias=False)),
                ('bn', nn.BatchNorm2d(out_channels)),
            ]))

    def forward(self, x):
        if self.inverted_bottleneck:
            x = self.inverted_bottleneck(x)
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x

    @staticmethod
    def is_zero_layer():
        return False


def conv_func_by_name(name):
    name2ops = {
        'Identity': lambda in_C, out_C, S: IdentityLayer(),
        'Zero': lambda in_C, out_C, S: ZeroLayer(stride=S),
    }
    name2ops.update({
        '3x3_MBConv1':
        lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 3, S, 1),
        '3x3_MBConv2':
        lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 3, S, 2),
        '3x3_MBConv3':
        lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 3, S, 3),
        '3x3_MBConv4':
        lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 3, S, 4),
        '3x3_MBConv5':
        lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 3, S, 5),
        '3x3_MBConv6':
        lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 3, S, 6),
        #######################################################################################
        '5x5_MBConv1':
        lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 5, S, 1),
        '5x5_MBConv2':
        lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 5, S, 2),
        '5x5_MBConv3':
        lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 5, S, 3),
        '5x5_MBConv4':
        lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 5, S, 4),
        '5x5_MBConv5':
        lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 5, S, 5),
        '5x5_MBConv6':
        lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 5, S, 6),
        #######################################################################################
        '7x7_MBConv1':
        lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 7, S, 1),
        '7x7_MBConv2':
        lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 7, S, 2),
        '7x7_MBConv3':
        lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 7, S, 3),
        '7x7_MBConv4':
        lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 7, S, 4),
        '7x7_MBConv5':
        lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 7, S, 5),
        '7x7_MBConv6':
        lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 7, S, 6),
    })
    return name2ops[name]


def build_candidate_ops(candidate_ops, in_channels, out_channels, stride,
                        ops_order):
    if candidate_ops is None:
        raise ValueError('please specify a candidate set')

    name2ops = {
        'Identity':
        lambda in_C, out_C, S: IdentityLayer(in_C, out_C, ops_order=ops_order),
        'Zero':
        lambda in_C, out_C, S: ZeroLayer(stride=S),
    }
    # add MBConv layers
    name2ops.update({
        '3x3_MBConv1':
        lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 3, S, 1),
        '3x3_MBConv2':
        lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 3, S, 2),
        '3x3_MBConv3':
        lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 3, S, 3),
        '3x3_MBConv4':
        lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 3, S, 4),
        '3x3_MBConv5':
        lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 3, S, 5),
        '3x3_MBConv6':
        lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 3, S, 6),
        #######################################################################################
        '5x5_MBConv1':
        lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 5, S, 1),
        '5x5_MBConv2':
        lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 5, S, 2),
        '5x5_MBConv3':
        lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 5, S, 3),
        '5x5_MBConv4':
        lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 5, S, 4),
        '5x5_MBConv5':
        lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 5, S, 5),
        '5x5_MBConv6':
        lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 5, S, 6),
        #######################################################################################
        '7x7_MBConv1':
        lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 7, S, 1),
        '7x7_MBConv2':
        lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 7, S, 2),
        '7x7_MBConv3':
        lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 7, S, 3),
        '7x7_MBConv4':
        lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 7, S, 4),
        '7x7_MBConv5':
        lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 7, S, 5),
        '7x7_MBConv6':
        lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 7, S, 6),
    })

    return [
        name2ops[name](in_channels, out_channels, stride)
        for name in candidate_ops
    ]


class MobileInvertedResidualBlock(nn.Module):

    def __init__(self, mobile_inverted_conv, shortcut):
        super(MobileInvertedResidualBlock, self).__init__()
        self.mobile_inverted_conv = mobile_inverted_conv
        self.shortcut = shortcut

    def forward(self, x):
        if self.mobile_inverted_conv.is_zero_layer():
            res = x
        elif self.shortcut is None or self.shortcut.is_zero_layer():
            res = self.mobile_inverted_conv(x)
        else:
            conv_x = self.mobile_inverted_conv(x)
            skip_x = self.shortcut(x)
            res = skip_x + conv_x
        return res


class AutoSTREncoder(nn.Module):

    def __init__(self,
                 in_channels,
                 out_dim=256,
                 with_lstm=True,
                 stride_stages='[(2, 2), (2, 2), (2, 1), (2, 1), (2, 1)]',
                 n_cell_stages=[3, 3, 3, 3, 3],
                 conv_op_ids=[5, 5, 5, 5, 5, 5, 5, 6, 6, 5, 4, 3, 4, 6, 6],
                 **kwargs):
        super().__init__()
        self.first_conv = nn.Sequential(
            nn.Conv2d(in_channels,
                      32,
                      kernel_size=(3, 3),
                      stride=1,
                      padding=1,
                      bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        stride_stages = eval(stride_stages)
        width_stages = [32, 64, 128, 256, 512]
        conv_candidates = [
            '5x5_MBConv1', '5x5_MBConv3', '5x5_MBConv6', '3x3_MBConv1',
            '3x3_MBConv3', '3x3_MBConv6', 'Zero'
        ]

        assert len(conv_op_ids) == sum(n_cell_stages)
        blocks = []
        input_channel = 32
        for width, n_cell, s in zip(width_stages, n_cell_stages,
                                    stride_stages):
            for i in range(n_cell):
                if i == 0:
                    stride = s
                else:
                    stride = (1, 1)
                block_i = len(blocks)
                conv_op = conv_func_by_name(
                    conv_candidates[conv_op_ids[block_i]])(input_channel,
                                                           width, stride)
                if stride == (1, 1) and input_channel == width:
                    shortcut = IdentityLayer()
                else:
                    shortcut = None
                inverted_residual_block = MobileInvertedResidualBlock(
                    conv_op, shortcut)
                blocks.append(inverted_residual_block)
                input_channel = width
        self.out_channels = input_channel

        self.blocks = nn.ModuleList(blocks)

        # with_lstm = False
        self.with_lstm = with_lstm
        if with_lstm:
            self.rnn = nn.LSTM(input_channel,
                               out_dim // 2,
                               bidirectional=True,
                               num_layers=2,
                               batch_first=True)
            self.out_channels = out_dim

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.first_conv(x)
        for block in self.blocks:
            x = block(x)
        cnn_feat = x.squeeze(dim=2)
        cnn_feat = cnn_feat.transpose(2, 1)
        if self.with_lstm:
            rnn_feat, _ = self.rnn(cnn_feat)
            return rnn_feat
        else:
            return cnn_feat
