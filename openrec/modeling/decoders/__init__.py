import torch.nn as nn

__all__ = ['build_decoder']


def build_decoder(config):
    # rec head
    from .abinet_decoder import ABINetDecoder
    from .aster_decoder import ASTERDecoder
    from .cdistnet_decoder import CDistNetDecoder
    from .cppd_decoder import CPPDDecoder
    from .ctc2d_decoder import CTCDecoder2D
    from .ctc_decoder import CTCDecoder
    from .dan_decoder import DANDecoder
    from .igtr_decoder import IGTRDecoder
    from .lister_decoder import LISTERDecoder
    from .lpv_decoder import LPVDecoder
    from .mgp_decoder import MGPDecoder
    from .nrtr_decoder import NRTRDecoder
    from .parseq_decoder import PARSeqDecoder
    from .robustscanner_decoder import RobustScannerDecoder
    from .sar_decoder import SARDecoder
    from .smtr_decoder import SMTRDecoder
    from .smtr_decoder_nattn import SMTRDecoderNumAttn
    from .srn_decoder import SRNDecoder
    from .visionlan_decoder import VisionLANDecoder

    support_dict = [
        'CTCDecoder', 'NRTRDecoder', 'CPPDDecoder', 'ABINetDecoder',
        'CDistNetDecoder', 'VisionLANDecoder', 'PARSeqDecoder', 'IGTRDecoder',
        'SMTRDecoder', 'LPVDecoder', 'SARDecoder', 'RobustScannerDecoder',
        'SRNDecoder', 'ASTERDecoder', 'CTCDecoder2D', 'LISTERDecoder',
        'GTCDecoder', 'SMTRDecoderNumAttn', 'MGPDecoder', 'DANDecoder'
    ]

    module_name = config.pop('name')
    assert module_name in support_dict, Exception(
        'head only support {}'.format(support_dict))
    module_class = eval(module_name)(**config)
    return module_class


class GTCDecoder(nn.Module):

    def __init__(self,
                 in_channels,
                 gtc_decoder,
                 ctc_decoder,
                 detach=True,
                 infer_gtc=False,
                 out_channels=0,
                 **kwargs):
        super(GTCDecoder, self).__init__()
        self.detach = detach
        self.infer_gtc = infer_gtc
        if infer_gtc:
            gtc_decoder['out_channels'] = out_channels[0]
            ctc_decoder['out_channels'] = out_channels[1]
            gtc_decoder['in_channels'] = in_channels
            ctc_decoder['in_channels'] = in_channels
            self.gtc_decoder = build_decoder(gtc_decoder)
        else:
            ctc_decoder['in_channels'] = in_channels
            ctc_decoder['out_channels'] = out_channels
        self.ctc_decoder = build_decoder(ctc_decoder)

    def forward(self, x, data=None):
        ctc_pred = self.ctc_decoder(x.detach() if self.detach else x,
                                    data=data)
        if self.training or self.infer_gtc:
            gtc_pred = self.gtc_decoder(x.flatten(2).transpose(1, 2),
                                        data=data)
            return {'gtc_pred': gtc_pred, 'ctc_pred': ctc_pred}
        else:
            return ctc_pred


class GTCDecoderTwo(nn.Module):

    def __init__(self,
                 in_channels,
                 gtc_decoder,
                 ctc_decoder,
                 infer_gtc=False,
                 out_channels=0,
                 **kwargs):
        super(GTCDecoderTwo, self).__init__()
        self.infer_gtc = infer_gtc
        gtc_decoder['out_channels'] = out_channels[0]
        ctc_decoder['out_channels'] = out_channels[1]
        gtc_decoder['in_channels'] = in_channels
        ctc_decoder['in_channels'] = in_channels
        self.gtc_decoder = build_decoder(gtc_decoder)
        self.ctc_decoder = build_decoder(ctc_decoder)

    def forward(self, x, data=None):
        x_ctc, x_gtc = x
        ctc_pred = self.ctc_decoder(x_ctc, data=data)
        if self.training or self.infer_gtc:
            gtc_pred = self.gtc_decoder(x_gtc.flatten(2).transpose(1, 2),
                                        data=data)
            return {'gtc_pred': gtc_pred, 'ctc_pred': ctc_pred}
        else:
            return ctc_pred
