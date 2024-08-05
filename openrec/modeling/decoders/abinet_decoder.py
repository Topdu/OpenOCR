import torch
import torch.nn as nn
import torch.nn.functional as F

from openrec.modeling.decoders.nrtr_decoder import PositionalEncoding, TransformerBlock


class BCNLanguage(nn.Module):

    def __init__(
        self,
        d_model=512,
        nhead=8,
        num_layers=4,
        dim_feedforward=2048,
        dropout=0.0,
        max_length=25,
        detach=True,
        num_classes=37,
    ):
        super().__init__()
        self.d_model = d_model
        self.detach = detach
        self.max_length = max_length + 1

        self.proj = nn.Linear(num_classes, d_model, False)
        self.token_encoder = PositionalEncoding(dropout=0.1,
                                                dim=d_model,
                                                max_len=self.max_length)
        self.pos_encoder = PositionalEncoding(dropout=0,
                                              dim=d_model,
                                              max_len=self.max_length)
        self.decoder = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                attention_dropout_rate=dropout,
                residual_dropout_rate=dropout,
                with_self_attn=False,
                with_cross_attn=True,
            ) for i in range(num_layers)
        ])

        self.cls = nn.Linear(d_model, num_classes)

    def forward(self, tokens, lengths):
        """
        Args:
            tokens: (N, T, C) where T is length, N is batch size and C is classes number
            lengths: (N,)
        """
        if self.detach:
            tokens = tokens.detach()
        embed = self.proj(tokens)  # (N, T, E)
        embed = self.token_encoder(embed)  # (N, T, E)
        mask = _get_mask(lengths, self.max_length)  # (N, 1, T, T)
        zeros = embed.new_zeros(*embed.shape)
        qeury = self.pos_encoder(zeros)
        for decoder_layer in self.decoder:
            qeury = decoder_layer(qeury, embed, cross_mask=mask)
        output = qeury  # (N, T, E)

        logits = self.cls(output)  # (N, T, C)
        return output, logits


def encoder_layer(in_c, out_c, k=3, s=2, p=1):
    return nn.Sequential(nn.Conv2d(in_c, out_c, k, s, p),
                         nn.BatchNorm2d(out_c), nn.ReLU(True))


class DecoderUpsample(nn.Module):

    def __init__(self, in_c, out_c, k=3, s=1, p=1, mode='nearest') -> None:
        super().__init__()
        self.align_corners = None if mode == 'nearest' else True
        self.mode = mode
        # nn.Upsample(size=size, scale_factor=scale_factor, mode=mode, align_corners=align_corners),
        self.w = nn.Sequential(
            nn.Conv2d(in_c, out_c, k, s, p),
            nn.BatchNorm2d(out_c),
            nn.ReLU(True),
        )

    def forward(self, x, size):
        x = F.interpolate(x,
                          size=size,
                          mode=self.mode,
                          align_corners=self.align_corners)
        return self.w(x)


class PositionAttention(nn.Module):

    def __init__(self,
                 max_length,
                 in_channels=512,
                 num_channels=64,
                 mode='nearest',
                 **kwargs):
        super().__init__()
        self.max_length = max_length
        self.k_encoder = nn.Sequential(
            encoder_layer(in_channels, num_channels, s=(1, 2)),
            encoder_layer(num_channels, num_channels, s=(2, 2)),
            encoder_layer(num_channels, num_channels, s=(2, 2)),
            encoder_layer(num_channels, num_channels, s=(2, 2)),
        )
        self.k_decoder = nn.ModuleList([
            DecoderUpsample(num_channels, num_channels, mode=mode),
            DecoderUpsample(num_channels, num_channels, mode=mode),
            DecoderUpsample(num_channels, num_channels, mode=mode),
            DecoderUpsample(num_channels, in_channels, mode=mode),
        ])

        self.pos_encoder = PositionalEncoding(dropout=0,
                                              dim=in_channels,
                                              max_len=max_length)
        self.project = nn.Linear(in_channels, in_channels)

    def forward(self, x, query=None):
        N, E, H, W = x.size()
        k, v = x, x  # (N, E, H, W)

        # calculate key vector
        features = []
        size_decoder = []
        for i in range(0, len(self.k_encoder)):
            size_decoder.append(k.shape[2:])
            k = self.k_encoder[i](k)
            features.append(k)
        for i in range(0, len(self.k_decoder) - 1):
            k = self.k_decoder[i](k, size=size_decoder[-(i + 1)])
            k = k + features[len(self.k_decoder) - 2 - i]
        k = self.k_decoder[-1](k, size=size_decoder[0])  # (N, E, H, W)
        # calculate query vector
        # TODO q=f(q,k)
        zeros = x.new_zeros(
            (N, self.max_length, E)) if query is None else query  # (N, T, E)
        q = self.pos_encoder(zeros)  # (N, T, E)
        q = self.project(q)  # (N, T, E)

        # calculate attention
        attn_scores = torch.bmm(q, k.flatten(2, 3))  # (N, T, (H*W))
        attn_scores = attn_scores / (E**0.5)
        attn_scores = F.softmax(attn_scores, dim=-1)

        # (N, E, H, W) -> (N, H, W, E) -> (N, (H*W), E)
        v = v.permute(0, 2, 3, 1).view(N, -1, E)  # (N, (H*W), E)
        attn_vecs = torch.bmm(attn_scores, v)  # (N, T, E)
        return attn_vecs, attn_scores.view(N, -1, H, W)


class ABINetDecoder(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 nhead=8,
                 num_layers=3,
                 dim_feedforward=2048,
                 dropout=0.1,
                 max_length=25,
                 iter_size=3,
                 **kwargs):
        super().__init__()
        self.max_length = max_length + 1
        d_model = in_channels
        self.pos_encoder = PositionalEncoding(dropout=0.1, dim=d_model)
        self.encoder = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                attention_dropout_rate=dropout,
                residual_dropout_rate=dropout,
                with_self_attn=True,
                with_cross_attn=False,
            ) for _ in range(num_layers)
        ])
        self.decoder = PositionAttention(
            max_length=self.max_length,  # additional stop token
            in_channels=d_model,
            num_channels=d_model // 8,
            mode='nearest',
        )
        self.out_channels = out_channels
        self.cls = nn.Linear(d_model, self.out_channels)
        self.iter_size = iter_size
        if iter_size > 0:
            self.language = BCNLanguage(
                d_model=d_model,
                nhead=nhead,
                num_layers=4,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                max_length=max_length,
                num_classes=self.out_channels,
            )
            # alignment
            self.w_att_align = nn.Linear(2 * d_model, d_model)
            self.cls_align = nn.Linear(d_model, self.out_channels)

    def forward(self, x, data=None):
        # bs, c, h, w
        x = x.permute([0, 2, 3, 1])  # bs, h, w, c
        _, H, W, C = x.shape
        # assert H % 8 == 0 and W % 16 == 0, 'The height and width should be multiples of 8 and 16.'
        feature = x.flatten(1, 2)  # bs, h*w, c
        feature = self.pos_encoder(feature)  # bs, h*w, c
        for encoder_layer in self.encoder:
            feature = encoder_layer(feature)
        # bs, h*w, c
        feature = feature.reshape([-1, H, W, C]).permute(0, 3, 1,
                                                         2)  # bs, c, h, w
        v_feature, _ = self.decoder(feature)  # (bs[N], T, E)
        vis_logits = self.cls(v_feature)  # (bs[N], T, E)
        align_lengths = _get_length(vis_logits)
        align_logits = vis_logits
        all_l_res, all_a_res = [], []
        for _ in range(self.iter_size):
            tokens = F.softmax(align_logits, dim=-1)
            lengths = torch.clamp(
                align_lengths, 2,
                self.max_length)  # TODO: move to language model
            l_feature, l_logits = self.language(tokens, lengths)

            # alignment
            all_l_res.append(l_logits)
            fuse = torch.cat((l_feature, v_feature), -1)
            f_att = torch.sigmoid(self.w_att_align(fuse))
            output = f_att * v_feature + (1 - f_att) * l_feature
            align_logits = self.cls_align(output)

            align_lengths = _get_length(align_logits)
            all_a_res.append(align_logits)
        if self.training:
            return {
                'align': all_a_res,
                'lang': all_l_res,
                'vision': vis_logits
            }
        else:
            return F.softmax(align_logits, -1)


def _get_length(logit):
    """Greed decoder to obtain length from logit."""
    out = logit.argmax(dim=-1) == 0
    non_zero_mask = out.int() != 0
    mask_max_values, mask_max_indices = torch.max(non_zero_mask.int(), dim=-1)
    mask_max_indices[mask_max_values == 0] = -1
    out = mask_max_indices + 1
    return out


def _get_mask(length, max_length):
    """Generate a square mask for the sequence.

    The masked positions are filled with float('-inf'). Unmasked positions are
    filled with float(0.0).
    """
    length = length.unsqueeze(-1)
    N = length.size(0)
    grid = torch.arange(0, max_length, device=length.device).unsqueeze(0)
    zero_mask = torch.zeros([N, max_length],
                            dtype=torch.float32,
                            device=length.device)
    inf_mask = torch.full([N, max_length],
                          float('-inf'),
                          dtype=torch.float32,
                          device=length.device)
    diag_mask = torch.diag(
        torch.full([max_length],
                   float('-inf'),
                   dtype=torch.float32,
                   device=length.device),
        diagonal=0,
    )
    mask = torch.where(grid >= length, inf_mask, zero_mask)
    mask = mask.unsqueeze(1) + diag_mask
    return mask.unsqueeze(1)
