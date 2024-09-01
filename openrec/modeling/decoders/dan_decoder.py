import torch
import torch.nn as nn
import torch.nn.functional as F


class CAM(nn.Module):
    '''
    Convolutional Alignment Module
    '''

    # Current version only supports input whose size is a power of 2, such as 32, 64, 128 etc.
    # You can adapt it to any input size by changing the padding or stride.
    def __init__(self,
                 channels_list=[64, 128, 256, 512],
                 strides_list=[[2, 2], [1, 1], [1, 1]],
                 in_shape=[8, 32],
                 maxT=25,
                 depth=4,
                 num_channels=128):
        super(CAM, self).__init__()
        # cascade multiscale features
        fpn = []
        for i in range(1, len(channels_list)):
            fpn.append(
                nn.Sequential(
                    nn.Conv2d(channels_list[i - 1], channels_list[i], (3, 3),
                              (strides_list[i - 1][0], strides_list[i - 1][1]),
                              1), nn.BatchNorm2d(channels_list[i]),
                    nn.ReLU(True)))
        self.fpn = nn.Sequential(*fpn)
        # convolutional alignment
        # convs
        assert depth % 2 == 0, 'the depth of CAM must be a even number.'
        # in_shape = scales[-1]
        strides = []
        conv_ksizes = []
        deconv_ksizes = []
        h, w = in_shape[0], in_shape[1]
        for i in range(0, int(depth / 2)):
            stride = [2] if 2**(depth / 2 - i) <= h else [1]
            stride = stride + [2] if 2**(depth / 2 - i) <= w else stride + [1]
            strides.append(stride)
            conv_ksizes.append([3, 3])
            deconv_ksizes.append([_**2 for _ in stride])
        convs = [
            nn.Sequential(
                nn.Conv2d(channels_list[-1], num_channels,
                          tuple(conv_ksizes[0]), tuple(strides[0]),
                          (int((conv_ksizes[0][0] - 1) / 2),
                           int((conv_ksizes[0][1] - 1) / 2))),
                nn.BatchNorm2d(num_channels), nn.ReLU(True))
        ]
        for i in range(1, int(depth / 2)):
            convs.append(
                nn.Sequential(
                    nn.Conv2d(num_channels, num_channels,
                              tuple(conv_ksizes[i]), tuple(strides[i]),
                              (int((conv_ksizes[i][0] - 1) / 2),
                               int((conv_ksizes[i][1] - 1) / 2))),
                    nn.BatchNorm2d(num_channels), nn.ReLU(True)))
        self.convs = nn.Sequential(*convs)
        # deconvs
        deconvs = []
        for i in range(1, int(depth / 2)):
            deconvs.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        num_channels, num_channels,
                        tuple(deconv_ksizes[int(depth / 2) - i]),
                        tuple(strides[int(depth / 2) - i]),
                        (int(deconv_ksizes[int(depth / 2) - i][0] / 4.),
                         int(deconv_ksizes[int(depth / 2) - i][1] / 4.))),
                    nn.BatchNorm2d(num_channels), nn.ReLU(True)))
        deconvs.append(
            nn.Sequential(
                nn.ConvTranspose2d(num_channels, maxT, tuple(deconv_ksizes[0]),
                                   tuple(strides[0]),
                                   (int(deconv_ksizes[0][0] / 4.),
                                    int(deconv_ksizes[0][1] / 4.))),
                nn.Sigmoid()))
        self.deconvs = nn.Sequential(*deconvs)

    def forward(self, input):
        x = input[0]
        for i in range(0, len(self.fpn)):
            # print(self.fpn[i](x).shape, input[i+1].shape)
            x = self.fpn[i](x) + input[i + 1]
        conv_feats = []
        for i in range(0, len(self.convs)):
            x = self.convs[i](x)
            conv_feats.append(x)
        for i in range(0, len(self.deconvs) - 1):
            x = self.deconvs[i](x)
            x = x + conv_feats[len(conv_feats) - 2 - i]
        x = self.deconvs[-1](x)
        return x


class CAMSimp(nn.Module):

    def __init__(self, maxT=25, num_channels=128):
        super(CAMSimp, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(num_channels, maxT, 1, 1, 0),
                                  nn.Sigmoid())

    def forward(self, x):
        x = self.conv(x)
        return x


class DANDecoder(nn.Module):
    '''
    Decoupled Text Decoder
    '''

    def __init__(self,
                 out_channels,
                 in_channels,
                 use_cam=True,
                 max_len=25,
                 channels_list=[64, 128, 256, 512],
                 strides_list=[[2, 2], [1, 1], [1, 1]],
                 in_shape=[8, 32],
                 depth=4,
                 dropout=0.3,
                 **kwargs):
        super(DANDecoder, self).__init__()
        self.eos = 0
        self.bos = out_channels - 2
        self.ignore_index = out_channels - 1
        nchannel = in_channels
        self.nchannel = in_channels
        self.use_cam = use_cam
        if use_cam:
            self.cam = CAM(channels_list=channels_list,
                           strides_list=strides_list,
                           in_shape=in_shape,
                           maxT=max_len + 1,
                           depth=depth,
                           num_channels=nchannel)
        else:
            self.cam = CAMSimp(maxT=max_len + 1, num_channels=nchannel)
        self.pre_lstm = nn.LSTM(nchannel,
                                int(nchannel / 2),
                                bidirectional=True)
        self.rnn = nn.GRUCell(nchannel * 2, nchannel)
        self.generator = nn.Sequential(nn.Dropout(p=dropout),
                                       nn.Linear(nchannel, out_channels - 2))
        self.char_embeddings = nn.Embedding(out_channels,
                                            embedding_dim=in_channels,
                                            padding_idx=out_channels - 1)

    def forward(self, inputs, data=None):
        A = self.cam(inputs)
        if isinstance(inputs, list):
            feature = inputs[-1]
        else:
            feature = inputs
        nB, nC, nH, nW = feature.shape
        nT = A.shape[1]
        # Normalize
        A = A / A.view(nB, nT, -1).sum(2).view(nB, nT, 1, 1)
        # weighted sum
        C = feature.view(nB, 1, nC, nH, nW) * A.view(nB, nT, 1, nH, nW)
        C = C.view(nB, nT, nC, -1).sum(3).transpose(1, 0)  # T, B, C
        C, _ = self.pre_lstm(C)  # T, B, C
        C = F.dropout(C, p=0.3, training=self.training)
        if self.training:
            text = data[0]
            text_length = data[-1]
            nsteps = int(text_length.max())
            gru_res = torch.zeros_like(C)
            hidden = torch.zeros(nB, self.nchannel).type_as(C.data)
            prev_emb = self.char_embeddings(text[:, 0])
            for i in range(0, nsteps + 1):
                hidden = self.rnn(torch.cat((C[i, :, :], prev_emb), dim=1),
                                  hidden)
                gru_res[i, :, :] = hidden
                prev_emb = self.char_embeddings(text[:, i + 1])
            gru_res = self.generator(gru_res)
            return gru_res[:nsteps + 1, :, :].transpose(1, 0)
        else:
            gru_res = torch.zeros_like(C)
            hidden = torch.zeros(nB, self.nchannel).type_as(C.data)
            prev_emb = self.char_embeddings(
                torch.zeros(nB, dtype=torch.int64, device=feature.device) +
                self.bos)
            dec_seq = torch.full((nB, nT),
                                self.ignore_index,
                                dtype=torch.int64,
                                device=feature.get_device())
            
            for i in range(0, nT):
                hidden = self.rnn(torch.cat((C[i, :, :], prev_emb), dim=1),
                                  hidden)
                gru_res[i, :, :] = hidden
                mid_res = self.generator(hidden).argmax(-1)
                dec_seq[:, i] = mid_res.squeeze(0)
                if (dec_seq == self.eos).any(dim=-1).all():
                    break
                prev_emb = self.char_embeddings(mid_res)
            gru_res = self.generator(gru_res)
            return F.softmax(gru_res.transpose(1, 0), -1)
