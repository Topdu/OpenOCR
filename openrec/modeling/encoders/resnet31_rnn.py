import torch.nn as nn


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding."""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution."""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class AsterBlock(nn.Module):

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(AsterBlock, self).__init__()
        self.conv1 = conv1x1(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
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


class ResNet_ASTER(nn.Module):
    """For aster or crnn."""

    def __init__(self, in_channels, with_lstm=True, n_group=1):
        super(ResNet_ASTER, self).__init__()
        self.with_lstm = with_lstm
        self.n_group = n_group

        self.out_channels = 512
        if with_lstm:
            self.out_channels = 512

        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels,
                      32,
                      kernel_size=(3, 3),
                      stride=1,
                      padding=1,
                      bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True))

        self.inplanes = 32
        self.layer1 = self._make_layer(32, 3, [2, 2])  # [16, 50]
        self.layer2 = self._make_layer(64, 4, [2, 2])  # [8, 25]
        self.layer3 = self._make_layer(128, 6, [2, 1])  # [4, 25]
        self.layer4 = self._make_layer(256, 6, [2, 1])  # [2, 25]
        self.layer5 = self._make_layer(512, 3, [2, 1])  # [1, 25]

        if with_lstm:
            self.rnn = nn.LSTM(512,
                               256,
                               bidirectional=True,
                               num_layers=2,
                               batch_first=True)
            self.out_planes = 2 * 256
        else:
            self.out_planes = 512

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes, blocks, stride):
        downsample = None
        if stride != [1, 1] or self.inplanes != planes:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes, stride),
                                       nn.BatchNorm2d(planes))

        layers = []
        layers.append(AsterBlock(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(AsterBlock(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)

        cnn_feat = x5.squeeze(2)  # [N, c, w]
        cnn_feat = cnn_feat.transpose(2, 1).contiguous()
        if self.with_lstm:
            rnn_feat, _ = self.rnn(cnn_feat)
            return rnn_feat
        else:
            return cnn_feat
