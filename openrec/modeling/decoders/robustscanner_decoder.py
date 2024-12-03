import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RobustScannerDecoder(nn.Module):

    def __init__(
            self,
            out_channels,  # 90 + unknown + start + padding
            in_channels,
            enc_outchannles=128,
            hybrid_dec_rnn_layers=2,
            hybrid_dec_dropout=0,
            position_dec_rnn_layers=2,
            max_len=25,
            mask=True,
            encode_value=False,
            **kwargs):
        super(RobustScannerDecoder, self).__init__()

        start_idx = out_channels - 2
        padding_idx = out_channels - 1
        end_idx = 0
        # encoder module
        self.encoder = ChannelReductionEncoder(in_channels=in_channels,
                                               out_channels=enc_outchannles)
        self.max_text_length = max_len + 1
        self.mask = mask
        # decoder module
        self.decoder = Decoder(
            num_classes=out_channels,
            dim_input=in_channels,
            dim_model=enc_outchannles,
            hybrid_decoder_rnn_layers=hybrid_dec_rnn_layers,
            hybrid_decoder_dropout=hybrid_dec_dropout,
            position_decoder_rnn_layers=position_dec_rnn_layers,
            max_len=max_len + 1,
            start_idx=start_idx,
            mask=mask,
            padding_idx=padding_idx,
            end_idx=end_idx,
            encode_value=encode_value)

    def forward(self, inputs, data=None):
        '''
        data: [label, valid_ratio, 'length']
        '''
        out_enc = self.encoder(inputs)
        bs = out_enc.shape[0]
        valid_ratios = None
        word_positions = torch.arange(0,
                                      self.max_text_length,
                                      device=inputs.device).unsqueeze(0).tile(
                                          [bs, 1])

        if self.mask:
            valid_ratios = data[-1]

        if self.training:
            max_len = data[1].max()
            label = data[0][:, :1 + max_len]  # label
            final_out = self.decoder(inputs, out_enc, label, valid_ratios,
                                     word_positions[:, :1 + max_len])
        if not self.training:
            final_out = self.decoder(inputs,
                                     out_enc,
                                     label=None,
                                     valid_ratios=valid_ratios,
                                     word_positions=word_positions,
                                     train_mode=False)
        return final_out


class BaseDecoder(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

    def forward_train(self, feat, out_enc, targets, img_metas):
        raise NotImplementedError

    def forward_test(self, feat, out_enc, img_metas):
        raise NotImplementedError

    def forward(self,
                feat,
                out_enc,
                label=None,
                valid_ratios=None,
                word_positions=None,
                train_mode=True):
        self.train_mode = train_mode

        if train_mode:
            return self.forward_train(feat, out_enc, label, valid_ratios,
                                      word_positions)
        return self.forward_test(feat, out_enc, valid_ratios, word_positions)


class ChannelReductionEncoder(nn.Module):
    """Change the channel number with a one by one convoluational layer.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
    """

    def __init__(self, in_channels, out_channels, **kwargs):
        super(ChannelReductionEncoder, self).__init__()

        weight = torch.nn.Parameter(
            torch.nn.init.xavier_normal_(torch.empty(out_channels, in_channels,
                                                     1, 1),
                                         gain=1.0))
        self.layer = nn.Conv2d(in_channels,
                               out_channels,
                               kernel_size=1,
                               stride=1,
                               padding=0)

        use_xavier_normal = 1
        if use_xavier_normal:
            self.layer.weight = weight

    def forward(self, feat):
        """
        Args:
            feat (Tensor): Image features with the shape of
                :math:`(N, C_{in}, H, W)`.

        Returns:
            Tensor: A tensor of shape :math:`(N, C_{out}, H, W)`.
        """
        return self.layer(feat)


def masked_fill(x, mask, value):
    y = torch.full(x.shape, value, x.dtype)
    return torch.where(mask, y, x)


class DotProductAttentionLayer(nn.Module):

    def __init__(self, dim_model=None):
        super().__init__()

        self.scale = dim_model**-0.5 if dim_model is not None else 1.

    def forward(self, query, key, value, mask=None):

        query = query.permute(0, 2, 1)
        logits = query @ key * self.scale

        if mask is not None:
            n, seq_len = mask.size()
            mask = mask.view(n, 1, seq_len)
            logits = logits.masked_fill(mask, float('-inf'))

        weights = F.softmax(logits, dim=2)
        value = value.transpose(1, 2)
        glimpse = weights @ value
        glimpse = glimpse.permute(0, 2, 1).contiguous()
        return glimpse


class SequenceAttentionDecoder(BaseDecoder):
    """Sequence attention decoder for RobustScanner.

    RobustScanner: `RobustScanner: Dynamically Enhancing Positional Clues for
    Robust Text Recognition <https://arxiv.org/abs/2007.07542>`_

    Args:
        num_classes (int): Number of output classes :math:`C`.
        rnn_layers (int): Number of RNN layers.
        dim_input (int): Dimension :math:`D_i` of input vector ``feat``.
        dim_model (int): Dimension :math:`D_m` of the model. Should also be the
            same as encoder output vector ``out_enc``.
        max_seq_len (int): Maximum output sequence length :math:`T`.
        start_idx (int): The index of `<SOS>`.
        mask (bool): Whether to mask input features according to
            ``img_meta['valid_ratio']``.
        padding_idx (int): The index of `<PAD>`.
        dropout (float): Dropout rate.
        return_feature (bool): Return feature or logits as the result.
        encode_value (bool): Whether to use the output of encoder ``out_enc``
            as `value` of attention layer. If False, the original feature
            ``feat`` will be used.

    Warning:
        This decoder will not predict the final class which is assumed to be
        `<PAD>`. Therefore, its output size is always :math:`C - 1`. `<PAD>`
        is also ignored by loss as specified in
        :obj:`mmocr.models.textrecog.recognizer.EncodeDecodeRecognizer`.
    """

    def __init__(self,
                 num_classes=None,
                 rnn_layers=2,
                 dim_input=512,
                 dim_model=128,
                 max_seq_len=40,
                 start_idx=0,
                 mask=True,
                 padding_idx=None,
                 dropout=0,
                 return_feature=False,
                 encode_value=False):
        super().__init__()

        self.num_classes = num_classes
        self.dim_input = dim_input
        self.dim_model = dim_model
        self.return_feature = return_feature
        self.encode_value = encode_value
        self.max_seq_len = max_seq_len
        self.start_idx = start_idx
        self.mask = mask

        self.embedding = nn.Embedding(self.num_classes,
                                      self.dim_model,
                                      padding_idx=padding_idx)

        self.sequence_layer = nn.LSTM(input_size=dim_model,
                                      hidden_size=dim_model,
                                      num_layers=rnn_layers,
                                      batch_first=True,
                                      dropout=dropout)

        self.attention_layer = DotProductAttentionLayer()

        self.prediction = None
        if not self.return_feature:
            pred_num_classes = num_classes - 1
            self.prediction = nn.Linear(
                dim_model if encode_value else dim_input, pred_num_classes)

    def forward_train(self, feat, out_enc, targets, valid_ratios):
        """
        Args:
            feat (Tensor): Tensor of shape :math:`(N, D_i, H, W)`.
            out_enc (Tensor): Encoder output of shape
                :math:`(N, D_m, H, W)`.
            targets (Tensor): a tensor of shape :math:`(N, T)`. Each element is the index of a
                character.
            valid_ratios (Tensor): valid length ratio of img.
        Returns:
            Tensor: A raw logit tensor of shape :math:`(N, T, C-1)` if
            ``return_feature=False``. Otherwise it would be the hidden feature
            before the prediction projection layer, whose shape is
            :math:`(N, T, D_m)`.
        """

        tgt_embedding = self.embedding(targets)

        n, c_enc, h, w = out_enc.shape
        assert c_enc == self.dim_model
        _, c_feat, _, _ = feat.shape
        assert c_feat == self.dim_input
        _, len_q, c_q = tgt_embedding.shape
        assert c_q == self.dim_model
        assert len_q <= self.max_seq_len

        query, _ = self.sequence_layer(tgt_embedding)

        query = query.permute(0, 2, 1).contiguous()

        key = out_enc.view(n, c_enc, h * w)

        if self.encode_value:
            value = key
        else:
            value = feat.view(n, c_feat, h * w)

        mask = None
        if valid_ratios is not None:
            mask = query.new_zeros((n, h, w))
            for i, valid_ratio in enumerate(valid_ratios):
                valid_width = min(w, math.ceil(w * valid_ratio))
                mask[i, :, valid_width:] = 1
            mask = mask.bool()
            mask = mask.view(n, h * w)

        attn_out = self.attention_layer(query, key, value, mask)
        attn_out = attn_out.permute(0, 2, 1).contiguous()

        if self.return_feature:
            return attn_out

        out = self.prediction(attn_out)

        return out

    def forward_test(self, feat, out_enc, valid_ratios):
        """
        Args:
            feat (Tensor): Tensor of shape :math:`(N, D_i, H, W)`.
            out_enc (Tensor): Encoder output of shape
                :math:`(N, D_m, H, W)`.
            valid_ratios (Tensor): valid length ratio of img.

        Returns:
            Tensor: The output logit sequence tensor of shape
            :math:`(N, T, C-1)`.
        """
        batch_size = feat.shape[0]

        decode_sequence = (torch.ones((batch_size, self.max_seq_len),
                                      dtype=torch.int64,
                                      device=feat.device) * self.start_idx)

        outputs = []
        for i in range(self.max_seq_len):
            step_out = self.forward_test_step(feat, out_enc, decode_sequence,
                                              i, valid_ratios)
            outputs.append(step_out)
            max_idx = torch.argmax(step_out, dim=1, keepdim=False)
            if i < self.max_seq_len - 1:
                decode_sequence[:, i + 1] = max_idx

        outputs = torch.stack(outputs, 1)

        return outputs

    def forward_test_step(self, feat, out_enc, decode_sequence, current_step,
                          valid_ratios):
        """
        Args:
            feat (Tensor): Tensor of shape :math:`(N, D_i, H, W)`.
            out_enc (Tensor): Encoder output of shape
                :math:`(N, D_m, H, W)`.
            decode_sequence (Tensor): Shape :math:`(N, T)`. The tensor that
                stores history decoding result.
            current_step (int): Current decoding step.
            valid_ratios (Tensor): valid length ratio of img

        Returns:
            Tensor: Shape :math:`(N, C-1)`. The logit tensor of predicted
            tokens at current time step.
        """

        embed = self.embedding(decode_sequence)

        n, c_enc, h, w = out_enc.shape
        assert c_enc == self.dim_model
        _, c_feat, _, _ = feat.shape
        assert c_feat == self.dim_input
        _, _, c_q = embed.shape
        assert c_q == self.dim_model

        query, _ = self.sequence_layer(embed)
        query = query.transpose(1, 2)
        key = torch.reshape(out_enc, (n, c_enc, h * w))
        if self.encode_value:
            value = key
        else:
            value = torch.reshape(feat, (n, c_feat, h * w))

        mask = None
        if valid_ratios is not None:
            mask = query.new_zeros((n, h, w))
            for i, valid_ratio in enumerate(valid_ratios):
                valid_width = min(w, math.ceil(w * valid_ratio))
                mask[i, :, valid_width:] = 1
            mask = mask.bool()
            mask = mask.view(n, h * w)

        # [n, c, l]
        attn_out = self.attention_layer(query, key, value, mask)
        out = attn_out[:, :, current_step]

        if self.return_feature:
            return out

        out = self.prediction(out)
        out = F.softmax(out, dim=-1)

        return out


class PositionAwareLayer(nn.Module):

    def __init__(self, dim_model, rnn_layers=2):
        super().__init__()

        self.dim_model = dim_model

        self.rnn = nn.LSTM(input_size=dim_model,
                           hidden_size=dim_model,
                           num_layers=rnn_layers,
                           batch_first=True)

        self.mixer = nn.Sequential(
            nn.Conv2d(dim_model, dim_model, kernel_size=3, stride=1,
                      padding=1), nn.ReLU(True),
            nn.Conv2d(dim_model, dim_model, kernel_size=3, stride=1,
                      padding=1))

    def forward(self, img_feature):
        n, c, h, w = img_feature.shape
        rnn_input = img_feature.permute(0, 2, 3, 1).contiguous()
        rnn_input = rnn_input.view(n * h, w, c)
        rnn_output, _ = self.rnn(rnn_input)
        rnn_output = rnn_output.view(n, h, w, c)
        rnn_output = rnn_output.permute(0, 3, 1, 2).contiguous()

        out = self.mixer(rnn_output)
        return out


class PositionAttentionDecoder(BaseDecoder):
    """Position attention decoder for RobustScanner.

    RobustScanner: `RobustScanner: Dynamically Enhancing Positional Clues for
    Robust Text Recognition <https://arxiv.org/abs/2007.07542>`_

    Args:
        num_classes (int): Number of output classes :math:`C`.
        rnn_layers (int): Number of RNN layers.
        dim_input (int): Dimension :math:`D_i` of input vector ``feat``.
        dim_model (int): Dimension :math:`D_m` of the model. Should also be the
            same as encoder output vector ``out_enc``.
        max_seq_len (int): Maximum output sequence length :math:`T`.
        mask (bool): Whether to mask input features according to
            ``img_meta['valid_ratio']``.
        return_feature (bool): Return feature or logits as the result.
        encode_value (bool): Whether to use the output of encoder ``out_enc``
            as `value` of attention layer. If False, the original feature
            ``feat`` will be used.

    Warning:
        This decoder will not predict the final class which is assumed to be
        `<PAD>`. Therefore, its output size is always :math:`C - 1`. `<PAD>`
        is also ignored by loss
    """

    def __init__(self,
                 num_classes=None,
                 rnn_layers=2,
                 dim_input=512,
                 dim_model=128,
                 max_seq_len=40,
                 mask=True,
                 return_feature=False,
                 encode_value=False):
        super().__init__()

        self.num_classes = num_classes
        self.dim_input = dim_input
        self.dim_model = dim_model
        self.max_seq_len = max_seq_len
        self.return_feature = return_feature
        self.encode_value = encode_value
        self.mask = mask

        self.embedding = nn.Embedding(self.max_seq_len + 1, self.dim_model)

        self.position_aware_module = PositionAwareLayer(
            self.dim_model, rnn_layers)

        self.attention_layer = DotProductAttentionLayer()

        self.prediction = None
        if not self.return_feature:
            pred_num_classes = num_classes - 1
            self.prediction = nn.Linear(
                dim_model if encode_value else dim_input, pred_num_classes)

    def _get_position_index(self, length, batch_size):
        position_index_list = []
        for i in range(batch_size):
            position_index = torch.range(0, length, step=1, dtype='int64')
            position_index_list.append(position_index)
        batch_position_index = torch.stack(position_index_list, dim=0)
        return batch_position_index

    def forward_train(self, feat, out_enc, targets, valid_ratios,
                      position_index):
        """
        Args:
            feat (Tensor): Tensor of shape :math:`(N, D_i, H, W)`.
            out_enc (Tensor): Encoder output of shape
                :math:`(N, D_m, H, W)`.
            targets (dict): A dict with the key ``padded_targets``, a
                tensor of shape :math:`(N, T)`. Each element is the index of a
                character.
            valid_ratios (Tensor): valid length ratio of img.
            position_index (Tensor): The position of each word.

        Returns:
            Tensor: A raw logit tensor of shape :math:`(N, T, C-1)` if
            ``return_feature=False``. Otherwise it will be the hidden feature
            before the prediction projection layer, whose shape is
            :math:`(N, T, D_m)`.
        """
        n, c_enc, h, w = out_enc.shape
        assert c_enc == self.dim_model
        _, c_feat, _, _ = feat.shape
        assert c_feat == self.dim_input
        _, len_q = targets.shape
        assert len_q <= self.max_seq_len

        position_out_enc = self.position_aware_module(out_enc)

        query = self.embedding(position_index)
        query = query.permute(0, 2, 1).contiguous()
        key = position_out_enc.view(n, c_enc, h * w)
        if self.encode_value:
            value = out_enc.view(n, c_enc, h * w)
        else:
            value = feat.view(n, c_feat, h * w)

        mask = None
        if valid_ratios is not None:
            mask = query.new_zeros((n, h, w))
            for i, valid_ratio in enumerate(valid_ratios):
                valid_width = min(w, math.ceil(w * valid_ratio))
                mask[i, :, valid_width:] = 1
            mask = mask.bool()
            mask = mask.view(n, h * w)

        attn_out = self.attention_layer(query, key, value, mask)
        attn_out = attn_out.permute(0, 2, 1).contiguous()

        if self.return_feature:
            return attn_out

        return self.prediction(attn_out)

    def forward_test(self, feat, out_enc, valid_ratios, position_index):
        """
        Args:
            feat (Tensor): Tensor of shape :math:`(N, D_i, H, W)`.
            out_enc (Tensor): Encoder output of shape
                :math:`(N, D_m, H, W)`.
            valid_ratios (Tensor): valid length ratio of img
            position_index (Tensor): The position of each word.

        Returns:
            Tensor: A raw logit tensor of shape :math:`(N, T, C-1)` if
            ``return_feature=False``. Otherwise it would be the hidden feature
            before the prediction projection layer, whose shape is
            :math:`(N, T, D_m)`.
        """
        n, c_enc, h, w = out_enc.shape
        assert c_enc == self.dim_model
        _, c_feat, _, _ = feat.shape
        assert c_feat == self.dim_input

        position_out_enc = self.position_aware_module(out_enc)

        query = self.embedding(position_index)
        query = query.permute(0, 2, 1).contiguous()
        key = position_out_enc.view(n, c_enc, h * w)
        if self.encode_value:
            value = torch.reshape(out_enc, (n, c_enc, h * w))
        else:
            value = torch.reshape(feat, (n, c_feat, h * w))

        mask = None
        if valid_ratios is not None:
            mask = query.new_zeros((n, h, w))
            for i, valid_ratio in enumerate(valid_ratios):
                valid_width = min(w, math.ceil(w * valid_ratio))
                mask[i, :, valid_width:] = 1
            mask = mask.bool()
            mask = mask.view(n, h * w)

        attn_out = self.attention_layer(query, key, value, mask)
        attn_out = attn_out.transpose(1, 2)  # [n, len_q, dim_v]

        if self.return_feature:
            return attn_out

        return self.prediction(attn_out)


class RobustScannerFusionLayer(nn.Module):

    def __init__(self, dim_model, dim=-1):
        super(RobustScannerFusionLayer, self).__init__()

        self.dim_model = dim_model
        self.dim = dim
        self.linear_layer = nn.Linear(dim_model * 2, dim_model * 2)

    def forward(self, x0, x1):
        assert x0.shape == x1.shape
        fusion_input = torch.concat((x0, x1), self.dim)
        output = self.linear_layer(fusion_input)
        output = F.glu(output, self.dim)

        return output


class Decoder(BaseDecoder):
    """Decoder for RobustScanner.

    RobustScanner: `RobustScanner: Dynamically Enhancing Positional Clues for
    Robust Text Recognition <https://arxiv.org/abs/2007.07542>`_

    Args:
        num_classes (int): Number of output classes :math:`C`.
        dim_input (int): Dimension :math:`D_i` of input vector ``feat``.
        dim_model (int): Dimension :math:`D_m` of the model. Should also be the
            same as encoder output vector ``out_enc``.
        max_seq_len (int): Maximum output sequence length :math:`T`.
        start_idx (int): The index of `<SOS>`.
        mask (bool): Whether to mask input features according to
            ``img_meta['valid_ratio']``.
        padding_idx (int): The index of `<PAD>`.
        encode_value (bool): Whether to use the output of encoder ``out_enc``
            as `value` of attention layer. If False, the original feature
            ``feat`` will be used.

    Warning:
        This decoder will not predict the final class which is assumed to be
        `<PAD>`. Therefore, its output size is always :math:`C - 1`. `<PAD>`
        is also ignored by loss as specified in
        :obj:`mmocr.models.textrecog.recognizer.EncodeDecodeRecognizer`.
    """

    def __init__(self,
                 num_classes=None,
                 dim_input=512,
                 dim_model=128,
                 hybrid_decoder_rnn_layers=2,
                 hybrid_decoder_dropout=0,
                 position_decoder_rnn_layers=2,
                 max_len=40,
                 start_idx=0,
                 mask=True,
                 padding_idx=None,
                 end_idx=0,
                 encode_value=False):
        super().__init__()
        self.num_classes = num_classes
        self.dim_input = dim_input
        self.dim_model = dim_model
        self.max_seq_len = max_len
        self.encode_value = encode_value
        self.start_idx = start_idx
        self.padding_idx = padding_idx
        self.end_idx = end_idx
        self.mask = mask

        # init hybrid decoder
        self.hybrid_decoder = SequenceAttentionDecoder(
            num_classes=num_classes,
            rnn_layers=hybrid_decoder_rnn_layers,
            dim_input=dim_input,
            dim_model=dim_model,
            max_seq_len=max_len,
            start_idx=start_idx,
            mask=mask,
            padding_idx=padding_idx,
            dropout=hybrid_decoder_dropout,
            encode_value=encode_value,
            return_feature=True)

        # init position decoder
        self.position_decoder = PositionAttentionDecoder(
            num_classes=num_classes,
            rnn_layers=position_decoder_rnn_layers,
            dim_input=dim_input,
            dim_model=dim_model,
            max_seq_len=max_len,
            mask=mask,
            encode_value=encode_value,
            return_feature=True)

        self.fusion_module = RobustScannerFusionLayer(
            self.dim_model if encode_value else dim_input)

        pred_num_classes = num_classes
        self.prediction = nn.Linear(dim_model if encode_value else dim_input,
                                    pred_num_classes)

    def forward_train(self, feat, out_enc, target, valid_ratios,
                      word_positions):
        """
        Args:
            feat (Tensor): Tensor of shape :math:`(N, D_i, H, W)`.
            out_enc (Tensor): Encoder output of shape
                :math:`(N, D_m, H, W)`.
            target (dict): A dict with the key ``padded_targets``, a
                tensor of shape :math:`(N, T)`. Each element is the index of a
                character.
            valid_ratios (Tensor):
            word_positions (Tensor): The position of each word.

        Returns:
            Tensor: A raw logit tensor of shape :math:`(N, T, C-1)`.
        """

        hybrid_glimpse = self.hybrid_decoder.forward_train(
            feat, out_enc, target, valid_ratios)
        position_glimpse = self.position_decoder.forward_train(
            feat, out_enc, target, valid_ratios, word_positions)

        fusion_out = self.fusion_module(hybrid_glimpse, position_glimpse)

        out = self.prediction(fusion_out)

        return out

    def forward_test(self, feat, out_enc, valid_ratios, word_positions):
        """
        Args:
            feat (Tensor): Tensor of shape :math:`(N, D_i, H, W)`.
            out_enc (Tensor): Encoder output of shape
                :math:`(N, D_m, H, W)`.
            valid_ratios (Tensor):
            word_positions (Tensor): The position of each word.
        Returns:
            Tensor: The output logit sequence tensor of shape
            :math:`(N, T, C-1)`.
        """
        seq_len = self.max_seq_len
        batch_size = feat.shape[0]

        decode_sequence = (torch.ones(
            (batch_size, seq_len), dtype=torch.int64, device=feat.device) *
                           self.start_idx)

        position_glimpse = self.position_decoder.forward_test(
            feat, out_enc, valid_ratios, word_positions)

        outputs = []
        for i in range(seq_len):
            hybrid_glimpse_step = self.hybrid_decoder.forward_test_step(
                feat, out_enc, decode_sequence, i, valid_ratios)

            fusion_out = self.fusion_module(hybrid_glimpse_step,
                                            position_glimpse[:, i, :])

            char_out = self.prediction(fusion_out)
            char_out = F.softmax(char_out, -1)
            outputs.append(char_out)
            max_idx = torch.argmax(char_out, dim=1, keepdim=False)
            if i < seq_len - 1:
                decode_sequence[:, i + 1] = max_idx
                if (decode_sequence == self.end_idx).any(dim=-1).all():
                    break
        outputs = torch.stack(outputs, 1)

        return outputs
