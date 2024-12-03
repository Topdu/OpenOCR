import torch
import torch.nn as nn
import torch.nn.functional as F

from openrec.modeling.decoders.nrtr_decoder import Embeddings, PositionalEncoding, TransformerBlock  # , Beam
from openrec.modeling.decoders.visionlan_decoder import Transformer_Encoder


def generate_square_subsequent_mask(sz):
    r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
    Unmasked positions are filled with float(0.0).
    """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = (mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(
        mask == 1, float(0.0)))
    return mask


class SEM_Pre(nn.Module):

    def __init__(
        self,
        d_model=512,
        dst_vocab_size=40,
        residual_dropout_rate=0.1,
    ):
        super(SEM_Pre, self).__init__()
        self.embedding = Embeddings(d_model=d_model, vocab=dst_vocab_size)

        self.positional_encoding = PositionalEncoding(
            dropout=residual_dropout_rate,
            dim=d_model,
        )

    def forward(self, tgt):
        tgt = self.embedding(tgt)
        tgt = self.positional_encoding(tgt)
        tgt_mask = generate_square_subsequent_mask(tgt.shape[1]).to(tgt.device)
        return tgt, tgt_mask


class POS_Pre(nn.Module):

    def __init__(
        self,
        d_model=512,
    ):
        super(POS_Pre, self).__init__()
        self.pos_encoding = PositionalEncoding(
            dropout=0.1,
            dim=d_model,
        )
        self.linear1 = nn.Linear(d_model, d_model)
        self.linear2 = nn.Linear(d_model, d_model)

        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, tgt):
        pos = tgt.new_zeros(*tgt.shape)
        pos = self.pos_encoding(pos)

        pos2 = self.linear2(F.relu(self.linear1(pos)))
        pos = self.norm2(pos + pos2)
        return pos


class DSF(nn.Module):

    def __init__(self, d_model, fusion_num):
        super(DSF, self).__init__()
        self.w_att = nn.Linear(fusion_num * d_model, d_model)

    def forward(self, l_feature, v_feature):
        """
        Args:
            l_feature: (N, T, E) where T is length, N is batch size and d is dim of model
            v_feature: (N, T, E) shape the same as l_feature
            l_lengths: (N,)
            v_lengths: (N,)
        """
        f = torch.cat((l_feature, v_feature), dim=2)
        f_att = torch.sigmoid(self.w_att(f))
        output = f_att * v_feature + (1 - f_att) * l_feature

        return output


class MDCDP(nn.Module):
    r"""
    Multi-Domain CharacterDistance Perception
    """

    def __init__(self, d_model, n_head, d_inner, num_layers):
        super(MDCDP, self).__init__()

        self.num_layers = num_layers

        # step 1 SAE
        self.layers_pos = nn.ModuleList([
            TransformerBlock(d_model, n_head, d_inner)
            for _ in range(num_layers)
        ])

        # step 2 CBI:
        self.layers2 = nn.ModuleList([
            TransformerBlock(
                d_model,
                n_head,
                d_inner,
                with_self_attn=False,
                with_cross_attn=True,
            ) for _ in range(num_layers)
        ])
        self.layers3 = nn.ModuleList([
            TransformerBlock(
                d_model,
                n_head,
                d_inner,
                with_self_attn=False,
                with_cross_attn=True,
            ) for _ in range(num_layers)
        ])

        # step 3 :DSF
        self.dynamic_shared_fusion = DSF(d_model, 2)

    def forward(
        self,
        sem,
        vis,
        pos,
        tgt_mask=None,
        memory_mask=None,
    ):

        for i in range(self.num_layers):
            # ----------step 1 -----------: SAE: Self-Attention Enhancement
            pos = self.layers_pos[i](pos, self_mask=tgt_mask)

            # ----------step 2 -----------: CBI: Cross-Branch Interaction

            # CBI-V
            pos_vis = self.layers2[i](
                pos,
                vis,
                cross_mask=memory_mask,
            )

            # CBI-S
            pos_sem = self.layers3[i](
                pos,
                sem,
                cross_mask=tgt_mask,
            )

            # ----------step 3 -----------: DSF: Dynamic Shared Fusion
            pos = self.dynamic_shared_fusion(pos_vis, pos_sem)

        output = pos
        return output


class ConvBnRelu(nn.Module):
    # adapt padding for kernel_size change
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        conv=nn.Conv2d,
        stride=2,
        inplace=True,
    ):
        super().__init__()
        p_size = [int(k // 2) for k in kernel_size]
        # p_size = int(kernel_size//2)
        self.conv = conv(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=p_size,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class CDistNetDecoder(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 n_head=None,
                 num_encoder_blocks=3,
                 num_decoder_blocks=3,
                 beam_size=0,
                 max_len=25,
                 residual_dropout_rate=0.1,
                 add_conv=False,
                 **kwargs):
        super(CDistNetDecoder, self).__init__()
        dst_vocab_size = out_channels
        self.ignore_index = dst_vocab_size - 1
        self.bos = dst_vocab_size - 2
        self.eos = 0
        self.beam_size = beam_size
        self.max_len = max_len
        self.add_conv = add_conv
        d_model = in_channels
        dim_feedforward = d_model * 4
        n_head = n_head if n_head is not None else d_model // 32

        if add_conv:
            self.convbnrelu = ConvBnRelu(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=(1, 3),
                stride=(1, 2),
            )
        if num_encoder_blocks > 0:
            self.positional_encoding = PositionalEncoding(
                dropout=0.1,
                dim=d_model,
            )
            self.trans_encoder = Transformer_Encoder(
                n_layers=num_encoder_blocks,
                n_head=n_head,
                d_model=d_model,
                d_inner=dim_feedforward,
            )
        else:
            self.trans_encoder = None
        self.semantic_branch = SEM_Pre(
            d_model=d_model,
            dst_vocab_size=dst_vocab_size,
            residual_dropout_rate=residual_dropout_rate,
        )
        self.positional_branch = POS_Pre(d_model=d_model)

        self.mdcdp = MDCDP(d_model, n_head, dim_feedforward // 2,
                           num_decoder_blocks)
        self._reset_parameters()

        self.tgt_word_prj = nn.Linear(
            d_model, dst_vocab_size - 2,
            bias=False)  # We don't predict <bos> nor <pad>
        self.tgt_word_prj.weight.data.normal_(mean=0.0, std=d_model**-0.5)

    def forward(self, x, data=None):
        if self.add_conv:
            x = self.convbnrelu(x)
            # x = rearrange(x, "b c h w -> b (w h) c")
        x = x.flatten(2).transpose(1, 2)
        if self.trans_encoder is not None:
            x = self.positional_encoding(x)
            vis_feat = self.trans_encoder(x, src_mask=None)
        else:
            vis_feat = x
        if self.training:
            max_len = data[1].max()
            tgt = data[0][:, :1 + max_len]
            res = self.forward_train(vis_feat, tgt)
        else:
            if self.beam_size > 0:
                res = self.forward_beam(vis_feat)
            else:
                res = self.forward_test(vis_feat)
        return res

    def forward_train(self, vis_feat, tgt):
        sem_feat, sem_mask = self.semantic_branch(tgt)
        pos_feat = self.positional_branch(sem_feat)
        output = self.mdcdp(
            sem_feat,
            vis_feat,
            pos_feat,
            tgt_mask=sem_mask,
            memory_mask=None,
        )

        logit = self.tgt_word_prj(output)
        return logit

    def forward_test(self, vis_feat):
        bs = vis_feat.size(0)

        dec_seq = torch.full(
            (bs, self.max_len + 1),
            self.ignore_index,
            dtype=torch.int64,
            device=vis_feat.device,
        )
        dec_seq[:, 0] = self.bos
        logits = []
        for len_dec_seq in range(0, self.max_len):
            sem_feat, sem_mask = self.semantic_branch(dec_seq[:, :len_dec_seq +
                                                              1])
            pos_feat = self.positional_branch(sem_feat)
            output = self.mdcdp(
                sem_feat,
                vis_feat,
                pos_feat,
                tgt_mask=sem_mask,
                memory_mask=None,
            )

            dec_output = output[:, -1:, :]

            word_prob = F.softmax(self.tgt_word_prj(dec_output), dim=-1)
            logits.append(word_prob)
            if len_dec_seq < self.max_len:
                # greedy decode. add the next token index to the target input
                dec_seq[:, len_dec_seq + 1] = word_prob.squeeze(1).argmax(-1)
                # Efficient batch decoding: If all output words have at least one EOS token, end decoding.
                if (dec_seq == self.eos).any(dim=-1).all():
                    break
        logits = torch.cat(logits, dim=1)
        return logits

    def forward_beam(self, x):
        """Translation work in one batch."""
        # to do

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
