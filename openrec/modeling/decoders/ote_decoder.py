import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.init import ones_, trunc_normal_, zeros_

from .nrtr_decoder import TransformerBlock, Embeddings


class CPA(nn.Module):

    def __init__(self, dim, max_len=25):
        super(CPA, self).__init__()

        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.fc3 = nn.Linear(dim, dim)
        self.pos_embed = nn.Parameter(torch.zeros([1, max_len + 1, dim],
                                                  dtype=torch.float32),
                                      requires_grad=True)
        trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, feat):
        # feat: B, L, Dim
        feat = feat.mean(1).unsqueeze(1)  # B, 1, Dim
        x = self.fc1(feat) + self.pos_embed  # B max_len dim
        x = F.softmax(self.fc2(F.tanh(x)), -1)  # B max_len dim
        x = self.fc3(feat * x)  # B max_len dim
        return x


class ARDecoder(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        nhead=None,
        num_decoder_layers=6,
        max_len=25,
        attention_dropout_rate=0.0,
        residual_dropout_rate=0.1,
        scale_embedding=True,
    ):
        super(ARDecoder, self).__init__()
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
        self.pos_embed = nn.Parameter(torch.zeros([1, max_len + 1, d_model],
                                                  dtype=torch.float32),
                                      requires_grad=True)
        trunc_normal_(self.pos_embed, std=0.02)
        self.decoder = nn.ModuleList([
            TransformerBlock(
                d_model,
                nhead,
                dim_feedforward,
                attention_dropout_rate,
                residual_dropout_rate,
                with_self_attn=True,
                with_cross_attn=False,
            ) for i in range(num_decoder_layers)
        ])

        self.tgt_word_prj = nn.Linear(d_model,
                                      self.out_channels - 2,
                                      bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward_train(self, src, tgt):
        tgt = tgt[:, :-1]

        tgt = self.embedding(
            tgt) + src[:, :tgt.shape[1]] + self.pos_embed[:, :tgt.shape[1]]
        tgt_mask = self.generate_square_subsequent_mask(
            tgt.shape[1], device=src.get_device())

        for decoder_layer in self.decoder:
            tgt = decoder_layer(tgt, self_mask=tgt_mask)
        output = tgt
        logit = self.tgt_word_prj(output)
        return logit

    def forward(self, src, data=None):

        if self.training:
            max_len = data[1].max()
            tgt = data[0][:, :2 + max_len]
            res = self.forward_train(src, tgt)
        else:
            res = self.forward_test(src)
        return res

    def forward_test(self, src):
        bs = src.shape[0]
        src = src + self.pos_embed
        dec_seq = torch.full((bs, self.max_len + 1),
                             self.ignore_index,
                             dtype=torch.int64,
                             device=src.get_device())
        dec_seq[:, 0] = self.bos
        logits = []
        for len_dec_seq in range(0, self.max_len):
            dec_seq_embed = self.embedding(
                dec_seq[:, :len_dec_seq + 1])  # N dim 26+10 # </s>  012 a
            dec_seq_embed = dec_seq_embed + src[:, :len_dec_seq + 1]
            tgt_mask = self.generate_square_subsequent_mask(
                dec_seq_embed.shape[1], src.get_device())
            tgt = dec_seq_embed  # bs, 3, dim #bos, a, b, c, ... eos
            for decoder_layer in self.decoder:
                tgt = decoder_layer(tgt, self_mask=tgt_mask)
            dec_output = tgt
            dec_output = dec_output[:, -1:, :]
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


class OTEDecoder(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 max_len=25,
                 num_heads=None,
                 ar=False,
                 num_decoder_layers=1,
                 **kwargs):
        super(OTEDecoder, self).__init__()

        self.out_channels = out_channels - 2  # none + 26 + 10
        dim = in_channels
        self.dim = dim
        self.max_len = max_len + 1  # max_len + eos

        self.cpa = CPA(dim=dim, max_len=max_len)
        self.ar = ar
        if ar:
            self.ar_decoder = ARDecoder(in_channels=dim,
                                        out_channels=out_channels,
                                        nhead=num_heads,
                                        num_decoder_layers=num_decoder_layers,
                                        max_len=max_len)
        else:
            self.fc = nn.Linear(dim, self.out_channels)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            zeros_(m.bias)
            ones_(m.weight)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed'}

    def forward(self, x, data=None):
        x = self.cpa(x)
        if self.ar:
            return self.ar_decoder(x, data=data)
        logits = self.fc(x)  # B, 26, 37
        if self.training:
            logits = logits[:, :data[1].max() + 1]
        return logits
