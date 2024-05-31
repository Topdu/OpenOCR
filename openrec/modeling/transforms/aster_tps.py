import itertools
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def conv3x3_block(in_planes, out_planes, stride=1):
    """3x3 convolution with padding."""
    conv_layer = nn.Conv2d(in_planes,
                           out_planes,
                           kernel_size=3,
                           stride=1,
                           padding=1)

    block = nn.Sequential(
        conv_layer,
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True),
    )
    return block


class STNHead(nn.Module):

    def __init__(self, in_planes, num_ctrlpoints, activation='none'):
        super(STNHead, self).__init__()

        self.in_planes = in_planes
        self.num_ctrlpoints = num_ctrlpoints
        self.activation = activation
        self.stn_convnet = nn.Sequential(
            conv3x3_block(in_planes, 32),  # 32*64
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv3x3_block(32, 64),  # 16*32
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv3x3_block(64, 128),  # 8*16
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv3x3_block(128, 256),  # 4*8
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv3x3_block(256, 256),  # 2*4,
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv3x3_block(256, 256))  # 1*2

        self.stn_fc1 = nn.Sequential(nn.Linear(2 * 256, 512),
                                     nn.BatchNorm1d(512),
                                     nn.ReLU(inplace=True))
        self.stn_fc2 = nn.Linear(512, num_ctrlpoints * 2)

        self.init_weights(self.stn_convnet)
        self.init_weights(self.stn_fc1)
        self.init_stn(self.stn_fc2)

    def init_weights(self, module):
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.001)
                m.bias.data.zero_()

    def init_stn(self, stn_fc2):
        margin = 0.01
        sampling_num_per_side = int(self.num_ctrlpoints / 2)
        ctrl_pts_x = np.linspace(margin, 1. - margin, sampling_num_per_side)
        ctrl_pts_y_top = np.ones(sampling_num_per_side) * margin
        ctrl_pts_y_bottom = np.ones(sampling_num_per_side) * (1 - margin)
        ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
        ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
        ctrl_points = np.concatenate([ctrl_pts_top, ctrl_pts_bottom],
                                     axis=0).astype(np.float32)
        if self.activation == 'none':
            pass
        elif self.activation == 'sigmoid':
            ctrl_points = -np.log(1. / ctrl_points - 1.)
        stn_fc2.weight.data.zero_()
        stn_fc2.bias.data = torch.Tensor(ctrl_points).view(-1)

    def forward(self, x):
        x = self.stn_convnet(x)
        batch_size, _, h, w = x.size()
        x = x.view(batch_size, -1)
        img_feat = self.stn_fc1(x)
        x = self.stn_fc2(0.1 * img_feat)
        if self.activation == 'sigmoid':
            x = F.sigmoid(x)
        x = x.view(-1, self.num_ctrlpoints, 2)
        return x


def grid_sample(input, grid, canvas=None):
    output = F.grid_sample(input, grid)
    if canvas is None:
        return output
    else:
        input_mask = input.data.new(input.size()).fill_(1)
        output_mask = F.grid_sample(input_mask, grid)
        padded_output = output * output_mask + canvas * (1 - output_mask)
        return padded_output


# phi(x1, x2) = r^2 * log(r), where r = ||x1 - x2||_2
def compute_partial_repr(input_points, control_points):
    N = input_points.size(0)
    M = control_points.size(0)
    pairwise_diff = input_points.view(N, 1, 2) - control_points.view(1, M, 2)
    # original implementation, very slow
    # pairwise_dist = torch.sum(pairwise_diff ** 2, dim = 2) # square of distance
    pairwise_diff_square = pairwise_diff * pairwise_diff
    pairwise_dist = pairwise_diff_square[:, :, 0] + pairwise_diff_square[:, :,
                                                                         1]
    repr_matrix = 0.5 * pairwise_dist * torch.log(pairwise_dist)
    # fix numerical error for 0 * log(0), substitute all nan with 0
    mask = repr_matrix != repr_matrix
    repr_matrix.masked_fill_(mask, 0)
    return repr_matrix


# output_ctrl_pts are specified, according to our task.
def build_output_control_points(num_control_points, margins):
    margin_x, margin_y = margins
    num_ctrl_pts_per_side = num_control_points // 2
    ctrl_pts_x = np.linspace(margin_x, 1.0 - margin_x, num_ctrl_pts_per_side)
    ctrl_pts_y_top = np.ones(num_ctrl_pts_per_side) * margin_y
    ctrl_pts_y_bottom = np.ones(num_ctrl_pts_per_side) * (1.0 - margin_y)
    ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
    ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
    # ctrl_pts_top = ctrl_pts_top[1:-1,:]
    # ctrl_pts_bottom = ctrl_pts_bottom[1:-1,:]
    output_ctrl_pts_arr = np.concatenate([ctrl_pts_top, ctrl_pts_bottom],
                                         axis=0)
    output_ctrl_pts = torch.Tensor(output_ctrl_pts_arr)
    return output_ctrl_pts


class TPSSpatialTransformer(nn.Module):

    def __init__(
        self,
        output_image_size,
        num_control_points,
        margins,
    ):
        super(TPSSpatialTransformer, self).__init__()
        self.output_image_size = output_image_size
        self.num_control_points = num_control_points
        self.margins = margins

        self.target_height, self.target_width = output_image_size
        target_control_points = build_output_control_points(
            num_control_points, margins)
        N = num_control_points
        # N = N - 4

        # create padded kernel matrix
        forward_kernel = torch.zeros(N + 3, N + 3)
        target_control_partial_repr = compute_partial_repr(
            target_control_points, target_control_points)
        forward_kernel[:N, :N].copy_(target_control_partial_repr)
        forward_kernel[:N, -3].fill_(1)
        forward_kernel[-3, :N].fill_(1)
        forward_kernel[:N, -2:].copy_(target_control_points)
        forward_kernel[-2:, :N].copy_(target_control_points.transpose(0, 1))
        # compute inverse matrix
        inverse_kernel = torch.inverse(forward_kernel)

        # create target cordinate matrix
        HW = self.target_height * self.target_width
        target_coordinate = list(
            itertools.product(range(self.target_height),
                              range(self.target_width)))
        target_coordinate = torch.Tensor(target_coordinate)  # HW x 2
        Y, X = target_coordinate.split(1, dim=1)
        Y = Y / (self.target_height - 1)
        X = X / (self.target_width - 1)
        target_coordinate = torch.cat([X, Y],
                                      dim=1)  # convert from (y, x) to (x, y)
        target_coordinate_partial_repr = compute_partial_repr(
            target_coordinate, target_control_points)
        target_coordinate_repr = torch.cat([
            target_coordinate_partial_repr,
            torch.ones(HW, 1), target_coordinate
        ],
                                           dim=1)

        # register precomputed matrices
        self.register_buffer('inverse_kernel', inverse_kernel)
        self.register_buffer('padding_matrix', torch.zeros(3, 2))
        self.register_buffer('target_coordinate_repr', target_coordinate_repr)
        self.register_buffer('target_control_points', target_control_points)

    def forward(self, input, source_control_points):
        assert source_control_points.ndimension() == 3
        assert source_control_points.size(1) == self.num_control_points
        assert source_control_points.size(2) == 2
        batch_size = source_control_points.size(0)

        Y = torch.cat([
            source_control_points,
            self.padding_matrix.expand(batch_size, 3, 2)
        ], 1)
        mapping_matrix = torch.matmul(self.inverse_kernel, Y)
        source_coordinate = torch.matmul(self.target_coordinate_repr,
                                         mapping_matrix)

        grid = source_coordinate.view(-1, self.target_height,
                                      self.target_width, 2)
        grid = torch.clamp(
            grid, 0, 1)  # the source_control_points may be out of [0, 1].
        # the input to grid_sample is normalized [-1, 1], but what we get is [0, 1]
        grid = 2.0 * grid - 1.0
        output_maps = grid_sample(input, grid, canvas=None)
        return output_maps


class Aster_TPS(nn.Module):

    def __init__(
        self,
        in_channels,
        tps_inputsize=[32, 64],
        tps_outputsize=[32, 100],
        num_control_points=20,
        tps_margins=[0.05, 0.05],
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        #TODO
        self.out_channels = in_channels
        self.tps_inputsize = tps_inputsize
        self.num_control_points = num_control_points

        self.stn_head = STNHead(
            in_planes=3,
            num_ctrlpoints=num_control_points,
        )

        self.tps = TPSSpatialTransformer(
            output_image_size=tps_outputsize,
            num_control_points=num_control_points,
            margins=tps_margins,
        )

    def forward(self, img):
        stn_input = F.interpolate(img,
                                  self.tps_inputsize,
                                  mode='bilinear',
                                  align_corners=True)

        ctrl_points = self.stn_head(stn_input)

        img = self.tps(img, ctrl_points)

        return img
