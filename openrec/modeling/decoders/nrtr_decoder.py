import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from openrec.modeling.common import Mlp


class NRTRDecoder(nn.Module):
    """A transformer model. User is able to modify the attributes as needed.
    The architechture is based on the paper "Attention Is All You Need". Ashish
    Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N
    Gomez, Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you
    need. In Advances in Neural Information Processing Systems, pages
    6000-6010.

    Args:
        d_model: the number of expected features in the encoder/decoder inputs (default=512).
        nhead: the number of heads in the multiheadattention models (default=8).
        num_encoder_layers: the number of sub-encoder-layers in the encoder (default=6).
        num_decoder_layers: the number of sub-decoder-layers in the decoder (default=6).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        custom_encoder: custom encoder (default=None).
        custom_decoder: custom decoder (default=None).
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        nhead=None,
        num_encoder_layers=6,
        beam_size=0,
        num_decoder_layers=6,
        max_len=25,
        attention_dropout_rate=0.0,
        residual_dropout_rate=0.1,
        scale_embedding=True,
    ):
        super(NRTRDecoder, self).__init__()
        self.out_channels = out_channels
        self.ignore_index = out_channels - 1
        self.bos = out_channels - 2
        self.eos = 0
        self.max_len = max_len
        d_model = in_channels
        dim_feedforward = d_model * 4
        nhead = nhead if nhead is not None else d_model // 32
        self.embedding = Embeddings(
            d_model=d_model,
            vocab=self.out_channels,
            padding_idx=0,
            scale_embedding=scale_embedding,
        )
        self.positional_encoding = PositionalEncoding(
            dropout=residual_dropout_rate, dim=d_model)

        if num_encoder_layers > 0:
            self.encoder = nn.ModuleList([
                TransformerBlock(
                    d_model,
                    nhead,
                    dim_feedforward,
                    attention_dropout_rate,
                    residual_dropout_rate,
                    with_self_attn=True,
                    with_cross_attn=False,
                ) for i in range(num_encoder_layers)
            ])
        else:
            self.encoder = None

        self.decoder = nn.ModuleList([
            TransformerBlock(
                d_model,
                nhead,
                dim_feedforward,
                attention_dropout_rate,
                residual_dropout_rate,
                with_self_attn=True,
                with_cross_attn=True,
            ) for i in range(num_decoder_layers)
        ])

        self.beam_size = beam_size
        self.d_model = d_model
        self.nhead = nhead
        self.tgt_word_prj = nn.Linear(d_model,
                                      self.out_channels - 2,
                                      bias=False)
        w0 = np.random.normal(0.0, d_model**-0.5,
                              (d_model, self.out_channels - 2)).astype(
                                  np.float32)
        self.tgt_word_prj.weight.data = torch.from_numpy(w0.transpose())
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward_train(self, src, tgt):
        tgt = tgt[:, :-1]

        tgt = self.embedding(tgt)
        tgt = self.positional_encoding(tgt)
        tgt_mask = self.generate_square_subsequent_mask(
            tgt.shape[1], device=src.get_device())

        if self.encoder is not None:
            src = self.positional_encoding(src)
            for encoder_layer in self.encoder:
                src = encoder_layer(src)
            memory = src  # B N C
        else:
            memory = src  # B N C
        for decoder_layer in self.decoder:
            tgt = decoder_layer(tgt, memory, self_mask=tgt_mask)
        output = tgt
        logit = self.tgt_word_prj(output)
        return logit

    def forward(self, src, data=None):
        """Take in and process masked source/target sequences.
        Args:
            src: the sequence to the encoder (required).
            tgt: the sequence to the decoder (required).
        Shape:
            - src: :math:`(B, sN, C)`.
            - tgt: :math:`(B, tN, C)`.
        Examples:
            >>> output = transformer_model(src, tgt)
        """

        if self.training:
            max_len = data[1].max()
            tgt = data[0][:, :2 + max_len]
            res = self.forward_train(src, tgt)
        else:
            res = self.forward_test(src)
        return res

    def forward_test(self, src):
        bs = src.shape[0]
        if self.encoder is not None:
            src = self.positional_encoding(src)
            for encoder_layer in self.encoder:
                src = encoder_layer(src)
            memory = src  # B N C
        else:
            memory = src
        dec_seq = torch.full((bs, self.max_len + 1),
                             self.ignore_index,
                             dtype=torch.int64,
                             device=src.get_device())
        dec_seq[:, 0] = self.bos
        logits = []
        self.attn_maps = []
        for len_dec_seq in range(0, self.max_len):
            dec_seq_embed = self.embedding(
                dec_seq[:, :len_dec_seq + 1])  # N dim 26+10 # </s>  012 a
            dec_seq_embed = self.positional_encoding(dec_seq_embed)
            tgt_mask = self.generate_square_subsequent_mask(
                dec_seq_embed.shape[1], src.get_device())
            tgt = dec_seq_embed  # bs, 3, dim #bos, a, b, c, ... eos
            for decoder_layer in self.decoder:
                tgt = decoder_layer(tgt, memory, self_mask=tgt_mask)
            self.attn_maps.append(
                self.decoder[-1].cross_attn.attn_map[0][:, -1:, :])
            dec_output = tgt
            dec_output = dec_output[:, -1:, :]

            word_prob = F.softmax(self.tgt_word_prj(dec_output), dim=-1)
            logits.append(word_prob)
            if len_dec_seq < self.max_len:
                # greedy decode. add the next token index to the target input
                dec_seq[:, len_dec_seq + 1] = word_prob.squeeze().argmax(-1)
                # Efficient batch decoding: If all output words have at least one EOS token, end decoding.
                if (dec_seq == self.eos).any(dim=-1).all():
                    break
        logits = torch.cat(logits, dim=1)
        return logits

    def generate_square_subsequent_mask(self, sz, device):
        """Generate a square mask for the sequence.

        The masked positions are filled with float('-inf'). Unmasked positions
        are filled with float(0.0).
        """
        mask = torch.zeros([sz, sz], dtype=torch.float32)
        mask_inf = torch.triu(
            torch.full((sz, sz), dtype=torch.float32, fill_value=-torch.inf),
            diagonal=1,
        )
        mask = mask + mask_inf
        return mask.unsqueeze(0).unsqueeze(0).to(device)


class MultiheadAttention(nn.Module):

    def __init__(self, embed_dim, num_heads, dropout=0.0, self_attn=False):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert (self.head_dim * num_heads == self.embed_dim
                ), 'embed_dim must be divisible by num_heads'
        self.scale = self.head_dim**-0.5
        self.self_attn = self_attn
        if self_attn:
            self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        else:
            self.q = nn.Linear(embed_dim, embed_dim)
            self.kv = nn.Linear(embed_dim, embed_dim * 2)
        self.attn_drop = nn.Dropout(dropout)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key=None, attn_mask=None):
        B, qN = query.shape[:2]

        if self.self_attn:
            qkv = self.qkv(query)
            qkv = qkv.reshape(B, qN, 3, self.num_heads,
                              self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
        else:
            kN = key.shape[1]
            q = self.q(query)
            q = q.reshape(B, qN, self.num_heads, self.head_dim).transpose(1, 2)
            kv = self.kv(key)
            kv = kv.reshape(B, kN, 2, self.num_heads,
                            self.head_dim).permute(2, 0, 3, 1, 4)
            k, v = kv.unbind(0)

        attn = (q.matmul(k.transpose(2, 3))) * self.scale
        if attn_mask is not None:
            attn += attn_mask

        attn = F.softmax(attn, dim=-1)
        if not self.training:
            self.attn_map = attn
        attn = self.attn_drop(attn)

        x = (attn.matmul(v)).transpose(1, 2)
        x = x.reshape(B, qN, self.embed_dim)
        x = self.out_proj(x)

        return x


class TransformerBlock(nn.Module):

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        attention_dropout_rate=0.0,
        residual_dropout_rate=0.1,
        with_self_attn=True,
        with_cross_attn=False,
        epsilon=1e-5,
    ):
        super(TransformerBlock, self).__init__()
        self.with_self_attn = with_self_attn
        if with_self_attn:
            self.self_attn = MultiheadAttention(d_model,
                                                nhead,
                                                dropout=attention_dropout_rate,
                                                self_attn=with_self_attn)
            self.norm1 = nn.LayerNorm(d_model, eps=epsilon)
            self.dropout1 = nn.Dropout(residual_dropout_rate)
        self.with_cross_attn = with_cross_attn
        if with_cross_attn:
            self.cross_attn = MultiheadAttention(
                d_model, nhead, dropout=attention_dropout_rate
            )  # for self_attn of encoder or cross_attn of decoder
            self.norm2 = nn.LayerNorm(d_model, eps=epsilon)
            self.dropout2 = nn.Dropout(residual_dropout_rate)

        self.mlp = Mlp(
            in_features=d_model,
            hidden_features=dim_feedforward,
            act_layer=nn.ReLU,
            drop=residual_dropout_rate,
        )

        self.norm3 = nn.LayerNorm(d_model, eps=epsilon)

        self.dropout3 = nn.Dropout(residual_dropout_rate)

    def forward(self, tgt, memory=None, self_mask=None, cross_mask=None):
        if self.with_self_attn:
            tgt1 = self.self_attn(tgt, attn_mask=self_mask)
            tgt = self.norm1(tgt + self.dropout1(tgt1))

        if self.with_cross_attn:
            tgt2 = self.cross_attn(tgt, key=memory, attn_mask=cross_mask)
            tgt = self.norm2(tgt + self.dropout2(tgt2))
        tgt = self.norm3(tgt + self.dropout3(self.mlp(tgt)))
        return tgt


class PositionalEncoding(nn.Module):
    """Inject some information about the relative or absolute position of the
    tokens in the sequence. The positional encodings have the same dimension as
    the embeddings, so that the two can be summed. Here, we use sine and cosine
    functions of different frequencies.

    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, dropout, dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros([max_len, dim])
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = torch.unsqueeze(pe, 0)
        # pe = torch.permute(pe, [1, 0, 2])
        self.register_buffer('pe', pe)

    def forward(self, x):
        """Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """
        # x = x.permute([1, 0, 2])
        # x = x + self.pe[:x.shape[0], :]
        x = x + self.pe[:, :x.shape[1], :]
        return self.dropout(x)  # .permute([1, 0, 2])


class PositionalEncoding_2d(nn.Module):
    """Inject some information about the relative or absolute position of the
    tokens in the sequence. The positional encodings have the same dimension as
    the embeddings, so that the two can be summed. Here, we use sine and cosine
    functions of different frequencies.

    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, dropout, dim, max_len=5000):
        super(PositionalEncoding_2d, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros([max_len, dim])
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = torch.permute(torch.unsqueeze(pe, 0), [1, 0, 2])
        self.register_buffer('pe', pe)

        self.avg_pool_1 = nn.AdaptiveAvgPool2d((1, 1))
        self.linear1 = nn.Linear(dim, dim)
        self.linear1.weight.data.fill_(1.0)
        self.avg_pool_2 = nn.AdaptiveAvgPool2d((1, 1))
        self.linear2 = nn.Linear(dim, dim)
        self.linear2.weight.data.fill_(1.0)

    def forward(self, x):
        """Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """
        w_pe = self.pe[:x.shape[-1], :]
        w1 = self.linear1(self.avg_pool_1(x).squeeze()).unsqueeze(0)
        w_pe = w_pe * w1
        w_pe = torch.permute(w_pe, [1, 2, 0])
        w_pe = torch.unsqueeze(w_pe, 2)

        h_pe = self.pe[:x.shape[-2], :]
        w2 = self.linear2(self.avg_pool_2(x).squeeze()).unsqueeze(0)
        h_pe = h_pe * w2
        h_pe = torch.permute(h_pe, [1, 2, 0])
        h_pe = torch.unsqueeze(h_pe, 3)

        x = x + w_pe + h_pe
        x = torch.permute(
            torch.reshape(x,
                          [x.shape[0], x.shape[1], x.shape[2] * x.shape[3]]),
            [2, 0, 1],
        )

        return self.dropout(x)


class Embeddings(nn.Module):

    def __init__(self, d_model, vocab, padding_idx=None, scale_embedding=True):
        super(Embeddings, self).__init__()
        self.embedding = nn.Embedding(vocab, d_model, padding_idx=padding_idx)
        self.embedding.weight.data.normal_(mean=0.0, std=d_model**-0.5)
        self.d_model = d_model
        self.scale_embedding = scale_embedding

    def forward(self, x):
        if self.scale_embedding:
            x = self.embedding(x)
            return x * math.sqrt(self.d_model)
        return self.embedding(x)
