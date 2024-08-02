import torch
import torch.nn as nn
import torch.nn.functional as F

from .nrtr_decoder import Embeddings, TransformerBlock


class PVAM(nn.Module):

    def __init__(self,
                 in_channels,
                 char_num,
                 max_text_length,
                 num_heads,
                 hidden_dims,
                 dropout_rate=0):
        super(PVAM, self).__init__()
        self.char_num = char_num
        self.max_length = max_text_length
        self.num_heads = num_heads
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        #TODO
        self.emb = nn.Embedding(num_embeddings=256,
                                embedding_dim=hidden_dims,
                                sparse=False)
        self.drop_out = nn.Dropout(dropout_rate)
        self.feat_emb = nn.Linear(in_channels, in_channels)
        self.token_emb = nn.Embedding(max_text_length, in_channels)
        self.score = nn.Linear(in_channels, 1, bias=False)

    def feat_pos_mix(self, conv_features, encoder_word_pos, dropout_rate):
        #b h*w c
        pos_emb = self.emb(encoder_word_pos)
        # pos_emb = pos_emb.detach()
        enc_input = conv_features + pos_emb

        if dropout_rate:
            enc_input = self.drop_out(enc_input)

        return enc_input

    def forward(self, inputs):
        b, c, h, w = inputs.shape
        conv_features = inputs.view(-1, c, h * w)
        conv_features = conv_features.permute(0, 2, 1).contiguous()
        # b h*w c

        # transformer encoder
        b, t, c = conv_features.shape

        encoder_feat_pos = torch.arange(t, dtype=torch.long).to(inputs.device)

        enc_inputs = self.feat_pos_mix(conv_features, encoder_feat_pos,
                                       self.dropout_rate)

        inputs = self.feat_emb(enc_inputs)  # feat emb

        inputs = inputs.unsqueeze(1).expand(-1, self.max_length, -1, -1)

        # b maxlen h*w c

        tokens_pos = torch.arange(self.max_length,
                                  dtype=torch.long).to(inputs.device)
        tokens_pos = tokens_pos.unsqueeze(0).expand(b, -1)

        tokens_pos_emd = self.token_emb(tokens_pos)
        tokens_pos_emd = tokens_pos_emd.unsqueeze(2).expand(-1, -1, t, -1)
        # b maxlen h*w c

        attention_weight = torch.tanh(tokens_pos_emd + inputs)

        attention_weight = torch.squeeze(self.score(attention_weight),
                                         -1)  #b,25,256

        attention_weight = F.softmax(attention_weight, dim=-1)  #b,25,256

        pvam_features = torch.matmul(attention_weight, enc_inputs)

        return pvam_features


class GSRM(nn.Module):

    def __init__(self,
                 in_channel,
                 char_num,
                 max_len,
                 num_heads,
                 hidden_dims,
                 num_layers,
                 dropout_rate=0,
                 attention_dropout=0.1):
        super(GSRM, self).__init__()
        self.char_num = char_num
        self.max_len = max_len
        self.num_heads = num_heads

        self.cls_op = nn.Linear(in_channel, self.char_num)
        self.cls_final = nn.Linear(in_channel, self.char_num)

        self.word_emb = Embeddings(d_model=hidden_dims, vocab=char_num)
        self.pos_emb = nn.Embedding(char_num, hidden_dims)
        self.dropout_rate = dropout_rate
        self.emb_drop_out = nn.Dropout(dropout_rate)

        self.forward_self_attn = nn.ModuleList([
            TransformerBlock(
                d_model=hidden_dims,
                nhead=num_heads,
                attention_dropout_rate=attention_dropout,
                residual_dropout_rate=0.1,
                dim_feedforward=hidden_dims,
                with_self_attn=True,
                with_cross_attn=False,
            ) for i in range(num_layers)
        ])

        self.backward_self_attn = nn.ModuleList([
            TransformerBlock(
                d_model=hidden_dims,
                nhead=num_heads,
                attention_dropout_rate=attention_dropout,
                residual_dropout_rate=0.1,
                dim_feedforward=hidden_dims,
                with_self_attn=True,
                with_cross_attn=False,
            ) for i in range(num_layers)
        ])

    def _pos_emb(self, word_seq, pos, dropoutrate):
        """
        word_Seq: bsz len
        pos: bsz len
        """
        word_emb_seq = self.word_emb(word_seq)
        pos_emb_seq = self.pos_emb(pos)
        # pos_emb_seq = pos_emb_seq.detach()

        input_mix = word_emb_seq + pos_emb_seq
        if dropoutrate > 0:
            input_mix = self.emb_drop_out(input_mix)

        return input_mix

    def forward(self, inputs):

        bos_idx = self.char_num - 2
        eos_idx = self.char_num - 1
        b, t, c = inputs.size()  #b,25,512
        inputs = inputs.view(-1, c)
        cls_res = self.cls_op(inputs)  #b,25,n_class

        word_pred_PVAM = F.softmax(cls_res, dim=-1).argmax(-1)
        word_pred_PVAM = word_pred_PVAM.view(-1, t, 1)
        #b 25 1
        word1 = F.pad(word_pred_PVAM, [0, 0, 1, 0], 'constant', value=bos_idx)
        word_forward = word1[:, :-1, :].squeeze(-1)

        word_backward = word_pred_PVAM.squeeze(-1)

        #mask
        attn_mask_forward = torch.triu(
            torch.full((self.max_len, self.max_len),
                       dtype=torch.float32,
                       fill_value=-torch.inf),
            diagonal=1,
        ).to(inputs.device)
        attn_mask_forward = attn_mask_forward.unsqueeze(0).expand(
            self.num_heads, -1, -1)
        attn_mask_backward = torch.tril(
            torch.full((self.max_len, self.max_len),
                       dtype=torch.float32,
                       fill_value=-torch.inf),
            diagonal=-1,
        ).to(inputs.device)
        attn_mask_backward = attn_mask_backward.unsqueeze(0).expand(
            self.num_heads, -1, -1)

        #B,25

        pos = torch.arange(self.max_len, dtype=torch.long).to(inputs.device)
        pos = pos.unsqueeze(0).expand(b, -1)  #b,25

        word_front_mix = self._pos_emb(word_forward, pos, self.dropout_rate)
        word_backward_mix = self._pos_emb(word_backward, pos,
                                          self.dropout_rate)
        # b 25 emb_dim

        for attn_layer in self.forward_self_attn:
            word_front_mix = attn_layer(word_front_mix,
                                        self_mask=attn_mask_forward)

        for attn_layer in self.backward_self_attn:
            word_backward_mix = attn_layer(word_backward_mix,
                                           self_mask=attn_mask_backward)

        #b,25,emb_dim
        eos_emd = self.word_emb(torch.full(
            (1, ), eos_idx).to(inputs.device)).expand(b, 1, -1)
        word_backward_mix = torch.cat((word_backward_mix, eos_emd), dim=1)
        word_backward_mix = word_backward_mix[:, 1:, ]

        gsrm_features = word_front_mix + word_backward_mix

        gsrm_out = self.cls_final(gsrm_features)
        # torch.matmul(gsrm_features,
        #                         self.word_emb.embedding.weight.permute(1, 0))

        b, t, c = gsrm_out.size()
        #b,25,n_class
        gsrm_out = gsrm_out.view(-1, c).contiguous()

        return gsrm_features, cls_res, gsrm_out


class VSFD(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(VSFD, self).__init__()
        self.char_num = out_channels
        self.fc0 = nn.Linear(in_channels * 2, in_channels)
        self.fc1 = nn.Linear(in_channels, self.char_num)

    def forward(self, pvam_feature, gsrm_feature):
        _, t, c1 = pvam_feature.size()
        _, t, c2 = gsrm_feature.size()
        combine_featurs = torch.cat([pvam_feature, gsrm_feature], dim=-1)
        combine_featurs = combine_featurs.view(-1, c1 + c2).contiguous()
        atten = self.fc0(combine_featurs)
        atten = torch.sigmoid(atten)
        atten = atten.view(-1, t, c1)
        combine_featurs = atten * pvam_feature + (1 - atten) * gsrm_feature
        combine_featurs = combine_featurs.view(-1, c1).contiguous()
        out = self.fc1(combine_featurs)
        return out


class SRNDecoder(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden_dims,
                 num_decoder_layers=4,
                 max_text_length=25,
                 num_heads=8,
                 **kwargs):
        super(SRNDecoder, self).__init__()

        self.max_text_length = max_text_length
        self.num_heads = num_heads

        self.pvam = PVAM(in_channels=in_channels,
                         char_num=out_channels,
                         max_text_length=max_text_length,
                         num_heads=num_heads,
                         hidden_dims=hidden_dims,
                         dropout_rate=0.1)

        self.gsrm = GSRM(in_channel=in_channels,
                         char_num=out_channels,
                         max_len=max_text_length,
                         num_heads=num_heads,
                         num_layers=num_decoder_layers,
                         hidden_dims=hidden_dims)

        self.vsfd = VSFD(in_channels=in_channels, out_channels=out_channels)

    def forward(self, feat, data=None):
        # feat [B,512,8,32]

        pvam_feature = self.pvam(feat)

        gsrm_features, pvam_preds, gsrm_preds = self.gsrm(pvam_feature)

        vsfd_preds = self.vsfd(pvam_feature, gsrm_features)

        if not self.training:
            preds = F.softmax(vsfd_preds, dim=-1)
            return preds

        return [pvam_preds, gsrm_preds, vsfd_preds]
