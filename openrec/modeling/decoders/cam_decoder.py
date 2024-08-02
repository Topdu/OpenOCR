import torch.nn as nn

from .nrtr_decoder import NRTRDecoder


class CAMDecoder(nn.Module):

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
        super().__init__()

        self.decoder = NRTRDecoder(
            in_channels=in_channels,
            out_channels=out_channels,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            beam_size=beam_size,
            num_decoder_layers=num_decoder_layers,
            max_len=max_len,
            attention_dropout_rate=attention_dropout_rate,
            residual_dropout_rate=residual_dropout_rate,
            scale_embedding=scale_embedding,
        )

    def forward(self, x, data=None):
        dec_in = x['refined_feat']
        dec_output = self.decoder(dec_in, data=data)
        x['rec_output'] = dec_output
        if self.training:
            return x
        else:
            return dec_output
