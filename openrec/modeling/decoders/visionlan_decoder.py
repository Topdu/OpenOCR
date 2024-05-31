import torch
import torch.nn as nn

from openrec.modeling.decoders.nrtr_decoder import PositionalEncoding, TransformerBlock


class Transformer_Encoder(nn.Module):

    def __init__(
        self,
        n_layers=3,
        n_head=8,
        d_model=512,
        d_inner=2048,
        dropout=0.1,
        n_position=256,
    ):

        super(Transformer_Encoder, self).__init__()
        self.pe = PositionalEncoding(dropout=dropout,
                                     dim=d_model,
                                     max_len=n_position)
        self.layer_stack = nn.ModuleList([
            TransformerBlock(d_model, n_head, d_inner) for _ in range(n_layers)
        ])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, enc_output, src_mask):
        enc_output = self.pe(enc_output)  # position embeding
        for enc_layer in self.layer_stack:
            enc_output = enc_layer(enc_output, self_mask=src_mask)
        enc_output = self.layer_norm(enc_output)
        return enc_output


class PP_layer(nn.Module):

    def __init__(self, n_dim=512, N_max_character=25, n_position=256):
        super(PP_layer, self).__init__()
        self.character_len = N_max_character
        self.f0_embedding = nn.Embedding(N_max_character, n_dim)
        self.w0 = nn.Linear(N_max_character, n_position)
        self.wv = nn.Linear(n_dim, n_dim)
        self.we = nn.Linear(n_dim, N_max_character)
        self.active = nn.Tanh()
        self.softmax = nn.Softmax(dim=2)

    def forward(self, enc_output):
        reading_order = torch.arange(self.character_len,
                                     dtype=torch.long,
                                     device=enc_output.device)
        reading_order = reading_order.unsqueeze(0).expand(
            enc_output.shape[0], -1)  # (S,) -> (B, S)
        reading_order = self.f0_embedding(reading_order)  # b,25,512
        # calculate attention

        t = self.w0(reading_order.transpose(1, 2))  # b,512,256
        t = self.active(t.transpose(1, 2) + self.wv(enc_output))  # b,256,512
        t = self.we(t)  # b,256,25
        t = self.softmax(t.transpose(1, 2))  # b,25,256
        g_output = torch.bmm(t, enc_output)  # b,25,512
        return g_output


class Prediction(nn.Module):

    def __init__(
        self,
        n_dim=512,
        n_class=37,
        N_max_character=25,
        n_position=256,
    ):
        super(Prediction, self).__init__()
        self.pp = PP_layer(n_dim=n_dim,
                           N_max_character=N_max_character,
                           n_position=n_position)
        self.pp_share = PP_layer(n_dim=n_dim,
                                 N_max_character=N_max_character,
                                 n_position=n_position)
        self.w_vrm = nn.Linear(n_dim, n_class)  # output layer
        self.w_share = nn.Linear(n_dim, n_class)  # output layer
        self.nclass = n_class

    def forward(self, cnn_feature, f_res, f_sub, is_Train=False, use_mlm=True):
        if is_Train:
            if not use_mlm:
                g_output = self.pp(cnn_feature)  # b,25,512
                g_output = self.w_vrm(g_output)
                f_res = 0
                f_sub = 0
                return g_output, f_res, f_sub
            g_output = self.pp(cnn_feature)  # b,25,512
            f_res = self.pp_share(f_res)
            f_sub = self.pp_share(f_sub)
            g_output = self.w_vrm(g_output)
            f_res = self.w_share(f_res)
            f_sub = self.w_share(f_sub)
            return g_output, f_res, f_sub
        else:
            g_output = self.pp(cnn_feature)  # b,25,512
            g_output = self.w_vrm(g_output)
            return g_output


class MLM(nn.Module):
    """Architecture of MLM."""

    def __init__(
        self,
        n_dim=512,
        n_position=256,
        n_head=8,
        dim_feedforward=2048,
        max_text_length=25,
    ):
        super(MLM, self).__init__()
        self.MLM_SequenceModeling_mask = Transformer_Encoder(
            n_layers=2,
            n_head=n_head,
            d_model=n_dim,
            d_inner=dim_feedforward,
            n_position=n_position,
        )
        self.MLM_SequenceModeling_WCL = Transformer_Encoder(
            n_layers=1,
            n_head=n_head,
            d_model=n_dim,
            d_inner=dim_feedforward,
            n_position=n_position,
        )
        self.pos_embedding = nn.Embedding(max_text_length, n_dim)
        self.w0_linear = nn.Linear(1, n_position)
        self.wv = nn.Linear(n_dim, n_dim)
        self.active = nn.Tanh()
        self.we = nn.Linear(n_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input, label_pos):
        # transformer unit for generating mask_c
        feature_v_seq = self.MLM_SequenceModeling_mask(input, src_mask=None)
        # position embedding layer
        pos_emb = self.pos_embedding(label_pos.long())
        pos_emb = self.w0_linear(torch.unsqueeze(pos_emb,
                                                 dim=2)).transpose(1, 2)
        # fusion position embedding with features V & generate mask_c
        att_map_sub = self.active(pos_emb + self.wv(feature_v_seq))
        att_map_sub = self.we(att_map_sub)  # b,256,1
        att_map_sub = self.sigmoid(att_map_sub.transpose(1, 2))  # b,1,256
        # WCL
        # generate inputs for WCL
        f_res = input * (1 - att_map_sub.transpose(1, 2)
                         )  # second path with remaining string
        f_sub = input * (att_map_sub.transpose(1, 2)
                         )  # first path with occluded character
        # transformer units in WCL
        f_res = self.MLM_SequenceModeling_WCL(f_res, src_mask=None)
        f_sub = self.MLM_SequenceModeling_WCL(f_sub, src_mask=None)
        return f_res, f_sub, att_map_sub


class MLM_VRM(nn.Module):

    def __init__(
        self,
        n_layers=3,
        n_position=256,
        n_dim=512,
        n_head=8,
        dim_feedforward=2048,
        max_text_length=25,
        nclass=37,
    ):
        super(MLM_VRM, self).__init__()
        self.MLM = MLM(
            n_dim=n_dim,
            n_position=n_position,
            n_head=n_head,
            dim_feedforward=dim_feedforward,
            max_text_length=max_text_length,
        )
        self.SequenceModeling = Transformer_Encoder(
            n_layers=n_layers,
            n_head=n_head,
            d_model=n_dim,
            d_inner=dim_feedforward,
            n_position=n_position,
        )
        self.Prediction = Prediction(
            n_dim=n_dim,
            n_position=n_position,
            N_max_character=max_text_length + 1,
            n_class=nclass,
        )  # N_max_character = 1 eos + 25 characters
        self.nclass = nclass
        self.max_text_length = max_text_length

    def forward(self, input, label_pos, training_step, is_Train=False):
        nT = self.max_text_length

        b, c, h, w = input.shape
        input = input.reshape(b, c, -1)
        input = input.transpose(1, 2)

        if is_Train:
            if training_step == 'LF_1':
                f_res = 0
                f_sub = 0
                input = self.SequenceModeling(input, src_mask=None)
                text_pre, text_rem, text_mas = self.Prediction(input,
                                                               f_res,
                                                               f_sub,
                                                               is_Train=True,
                                                               use_mlm=False)
                return text_pre, text_pre, text_pre
            elif training_step == 'LF_2':
                # MLM
                f_res, f_sub, mask_c = self.MLM(input, label_pos)
                input = self.SequenceModeling(input, src_mask=None)
                text_pre, text_rem, text_mas = self.Prediction(input,
                                                               f_res,
                                                               f_sub,
                                                               is_Train=True)
                return text_pre, text_rem, text_mas
            elif training_step == 'LA':
                # MLM
                f_res, f_sub, mask_c = self.MLM(input, label_pos)
                # use the mask_c (1 for occluded character and 0 for remaining characters) to occlude input
                # ratio controls the occluded number in a batch
                ratio = 2
                character_mask = torch.zeros_like(mask_c)
                character_mask[0:b // ratio, :, :] = mask_c[0:b // ratio, :, :]
                input = input * (1 - character_mask.transpose(1, 2))
                # VRM
                # transformer unit for VRM
                input = self.SequenceModeling(input, src_mask=None)
                # prediction layer for MLM and VSR
                text_pre, text_rem, text_mas = self.Prediction(input,
                                                               f_res,
                                                               f_sub,
                                                               is_Train=True)
                return text_pre, text_rem, text_mas
        else:  # VRM is only used in the testing stage
            f_res = 0
            f_sub = 0
            contextual_feature = self.SequenceModeling(input, src_mask=None)
            C = self.Prediction(contextual_feature,
                                f_res,
                                f_sub,
                                is_Train=False,
                                use_mlm=False)
            C = C.transpose(1, 0)  # (25, b, 38))
            out_res = torch.zeros(nT, b, self.nclass).type_as(input.data)

            out_length = torch.zeros(b).type_as(input.data)
            now_step = 0
            while 0 in out_length and now_step < nT:
                tmp_result = C[now_step, :, :]
                out_res[now_step] = tmp_result
                tmp_result = tmp_result.topk(1)[1].squeeze(dim=1)
                for j in range(b):
                    if out_length[j] == 0 and tmp_result[j] == 0:
                        out_length[j] = now_step + 1
                now_step += 1
            for j in range(0, b):
                if int(out_length[j]) == 0:
                    out_length[j] = nT
            start = 0
            output = torch.zeros(int(out_length.sum()),
                                 self.nclass).type_as(input.data)
            for i in range(0, b):
                cur_length = int(out_length[i])
                output[start:start + cur_length] = out_res[0:cur_length, i, :]
                start += cur_length

            return output, out_length


class VisionLANDecoder(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        n_head=None,
        training_step='LA',
        n_layers=3,
        n_position=256,
        max_text_length=25,
    ):
        super(VisionLANDecoder, self).__init__()
        self.training_step = training_step
        n_dim = in_channels
        dim_feedforward = n_dim * 4
        n_head = n_head if n_head is not None else n_dim // 32

        self.MLM_VRM = MLM_VRM(
            n_layers=n_layers,
            n_position=n_position,
            n_dim=n_dim,
            n_head=n_head,
            dim_feedforward=dim_feedforward,
            max_text_length=max_text_length,
            nclass=out_channels + 1,
        )

    def forward(self, x, data=None):
        # MLM + VRM
        if self.training:
            label_pos = data[-2]
            text_pre, text_rem, text_mas = self.MLM_VRM(x,
                                                        label_pos,
                                                        self.training_step,
                                                        is_Train=True)
            return text_pre, text_rem, text_mas
        else:
            output, out_length = self.MLM_VRM(x,
                                              None,
                                              self.training_step,
                                              is_Train=False)
            return output, out_length
