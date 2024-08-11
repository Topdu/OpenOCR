from torch import nn


class NRTREncoder(nn.Module):

    def __init__(self, in_channels):
        super(NRTREncoder, self).__init__()
        self.out_channels = 512  # 64*H
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=1,
            ), nn.ReLU(), nn.BatchNorm2d(32),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=2,
                padding=1,
            ), nn.ReLU(), nn.BatchNorm2d(64))

    def forward(self, images):
        x = self.block(images)
        x = x.permute(0, 3, 2, 1).flatten(2)  # B, W, H*C
        return x
