import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SAREncoder(nn.Module):

    def __init__(self,
                 enc_bi_rnn=False,
                 enc_drop_rnn=0.1,
                 in_channels=512,
                 d_enc=512,
                 **kwargs):
        super().__init__()

        # LSTM Encoder
        if enc_bi_rnn:
            bidirectional = True
        else:
            bidirectional = False

        hidden_size = d_enc

        self.rnn_encoder = nn.LSTM(input_size=in_channels,
                                   hidden_size=hidden_size,
                                   num_layers=2,
                                   dropout=enc_drop_rnn,
                                   bidirectional=bidirectional,
                                   batch_first=True)

        # global feature transformation
        encoder_rnn_out_size = hidden_size * (int(enc_bi_rnn) + 1)
        self.linear = nn.Linear(encoder_rnn_out_size, encoder_rnn_out_size)

    def forward(self, feat):

        h_feat = feat.shape[2]
        feat_v = F.max_pool2d(feat,
                              kernel_size=(h_feat, 1),
                              stride=1,
                              padding=0)
        feat_v = feat_v.squeeze(2)
        feat_v = feat_v.permute(0, 2, 1).contiguous()  # bsz * W * C

        holistic_feat = self.rnn_encoder(feat_v)[0]  # bsz * T * hidden_size

        valid_hf = holistic_feat[:, -1, :]  # bsz * hidden_size

        holistic_feat = self.linear(valid_hf)  # bsz * C

        return holistic_feat


class SARDecoder(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 max_len=25,
                 enc_bi_rnn=False,
                 enc_drop_rnn=0.1,
                 dec_bi_rnn=False,
                 dec_drop_rnn=0.0,
                 pred_dropout=0.1,
                 pred_concat=True,
                 mask=True,
                 use_lstm=True,
                 **kwargs):
        super(SARDecoder, self).__init__()

        self.num_classes = out_channels
        self.start_idx = out_channels - 2
        self.padding_idx = out_channels - 1
        self.end_idx = 0
        self.max_seq_len = max_len + 1
        self.pred_concat = pred_concat
        self.mask = mask
        enc_dim = in_channels
        d = in_channels
        embedding_dim = in_channels
        dec_dim = in_channels
        self.use_lstm = use_lstm
        if use_lstm:
            # encoder module
            self.encoder = SAREncoder(enc_bi_rnn=enc_bi_rnn,
                                      enc_drop_rnn=enc_drop_rnn,
                                      in_channels=in_channels,
                                      d_enc=enc_dim)

        # decoder module

        # 2D attention layer
        self.conv1x1_1 = nn.Linear(dec_dim, d)
        self.conv3x3_1 = nn.Conv2d(in_channels,
                                   d,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)
        self.conv1x1_2 = nn.Linear(d, 1)

        # Decoder input embedding
        self.embedding = nn.Embedding(self.num_classes,
                                      embedding_dim,
                                      padding_idx=self.padding_idx)

        self.rnndecoder = nn.LSTM(input_size=embedding_dim,
                                  hidden_size=dec_dim,
                                  num_layers=2,
                                  dropout=dec_drop_rnn,
                                  bidirectional=dec_bi_rnn,
                                  batch_first=True)

        # Prediction layer
        self.pred_dropout = nn.Dropout(pred_dropout)
        if pred_concat:
            fc_in_channel = in_channels + in_channels + dec_dim
        else:
            fc_in_channel = in_channels
        self.prediction = nn.Linear(fc_in_channel, self.num_classes)
        self.softmax = nn.Softmax(dim=-1)

    def _2d_attation(self, feat, tokens, data, training):

        Hidden_state = self.rnndecoder(tokens)[0]
        attn_query = self.conv1x1_1(Hidden_state)
        bsz, seq_len, _ = attn_query.size()
        attn_query = attn_query.unsqueeze(-1).unsqueeze(-1)
        # bsz * seq_len+1 * attn_size * 1 * 1
        attn_key = self.conv3x3_1(feat).unsqueeze(1)
        # bsz * 1 * attn_size * h * w

        attn_weight = torch.tanh(torch.add(attn_key, attn_query, alpha=1))
        attn_weight = attn_weight.permute(0, 1, 3, 4, 2).contiguous()
        attn_weight = self.conv1x1_2(attn_weight)

        _, T, h, w, c = attn_weight.size()

        if self.mask:
            valid_ratios = data[-1]
            # cal mask of attention weight
            attn_mask = torch.zeros_like(attn_weight)
            for i, valid_ratio in enumerate(valid_ratios):
                valid_width = min(w, math.ceil(w * valid_ratio))
                attn_mask[i, :, :, valid_width:, :] = 1
            attn_weight = attn_weight.masked_fill(attn_mask.bool(),
                                                  float('-inf'))

        attn_weight = attn_weight.view(bsz, T, -1)
        attn_weight = F.softmax(attn_weight, dim=-1)
        attn_weight = attn_weight.view(bsz, T, h, w,
                                       c).permute(0, 1, 4, 2, 3).contiguous()
        # bsz, T, 1, h, w
        # bsz, 1, f_c ,h, w
        attn_feat = torch.sum(torch.mul(feat.unsqueeze(1), attn_weight),
                              (3, 4),
                              keepdim=False)
        return [Hidden_state, attn_feat]

    def forward_train(self, feat, holistic_feat, data):

        max_len = data[1].max()
        label = data[0][:, :1 + max_len]  # label
        label_embedding = self.embedding(label)
        holistic_feat = holistic_feat.unsqueeze(1)
        tokens = torch.cat((holistic_feat, label_embedding), dim=1)

        Hidden_state, attn_feat = self._2d_attation(feat,
                                                    tokens,
                                                    data,
                                                    training=self.training)

        bsz, seq_len, f_c = Hidden_state.size()
        # linear transformation
        if self.pred_concat:
            f_c = holistic_feat.size(-1)
            holistic_feat = holistic_feat.expand(bsz, seq_len, f_c)
            preds = self.prediction(
                torch.cat((Hidden_state, attn_feat, holistic_feat), 2))
        else:
            preds = self.prediction(attn_feat)
        # bsz * (seq_len + 1) * num_classes
        preds = self.pred_dropout(preds)
        return preds[:, 1:, :]

    def forward_test(self, feat, holistic_feat, data=None):
        bsz = feat.shape[0]
        seq_len = self.max_seq_len
        holistic_feat = holistic_feat.unsqueeze(1)
        tokens = torch.full((bsz, ),
                            self.start_idx,
                            device=feat.device,
                            dtype=torch.long)
        outputs = []
        tokens = self.embedding(tokens)
        tokens = tokens.unsqueeze(1).expand(-1, seq_len, -1)
        tokens = torch.cat((holistic_feat, tokens), dim=1)
        for i in range(1, seq_len + 1):
            Hidden_state, attn_feat = self._2d_attation(feat,
                                                        tokens,
                                                        data=data,
                                                        training=self.training)
            if self.pred_concat:
                f_c = holistic_feat.size(-1)
                holistic_feat = holistic_feat.expand(bsz, seq_len + 1, f_c)
                preds = self.prediction(
                    torch.cat((Hidden_state, attn_feat, holistic_feat), 2))
            else:
                preds = self.prediction(attn_feat)
            # bsz * (seq_len + 1) * num_classes
            char_output = preds[:, i, :]
            char_output = F.softmax(char_output, -1)
            outputs.append(char_output)
            _, max_idx = torch.max(char_output, dim=1, keepdim=False)
            char_embedding = self.embedding(max_idx)
            if (i < seq_len):
                tokens[:, i + 1, :] = char_embedding
                if (tokens == self.end_idx).any(dim=-1).all():
                    break
        outputs = torch.stack(outputs, 1)

        return outputs

    def forward(self, feat, data=None):
        if self.use_lstm:
            holistic_feat = self.encoder(feat)  # bsz c
        else:
            holistic_feat = F.adaptive_avg_pool2d(feat, (1, 1)).squeeze()

        if self.training:
            preds = self.forward_train(feat, holistic_feat, data=data)
        else:
            preds = self.forward_test(feat, holistic_feat, data=data)
            # (bsz, seq_len, num_classes)
        return preds
