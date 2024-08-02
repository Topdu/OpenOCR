import copy

__all__ = ['build_post_process']

from .abinet_postprocess import ABINetLabelDecode
from .ar_postprocess import ARLabelDecode
from .ce_postprocess import CELabelDecode
from .char_postprocess import CharLabelDecode
from .cppd_postprocess import CPPDLabelDecode
from .ctc_postprocess import CTCLabelDecode
from .igtr_postprocess import IGTRLabelDecode
from .lister_postprocess import LISTERLabelDecode
from .mgp_postprocess import MPGLabelDecode
from .nrtr_postprocess import NRTRLabelDecode
from .smtr_postprocess import SMTRLabelDecode
from .srn_postprocess import SRNLabelDecode
from .visionlan_postprocess import VisionLANLabelDecode

support_dict = [
    'CTCLabelDecode', 'CharLabelDecode', 'CELabelDecode', 'CPPDLabelDecode',
    'NRTRLabelDecode', 'ABINetLabelDecode', 'ARLabelDecode', 'IGTRLabelDecode',
    'VisionLANLabelDecode', 'SMTRLabelDecode', 'SRNLabelDecode',
    'LISTERLabelDecode', 'GTCLabelDecode', 'MPGLabelDecode'
]


def build_post_process(config, global_config=None):
    config = copy.deepcopy(config)
    module_name = config.pop('name')
    if global_config is not None:
        config.update(global_config)
    assert module_name in support_dict, Exception(
        'post process only support {}'.format(support_dict))
    module_class = eval(module_name)(**config)
    return module_class


class GTCLabelDecode(object):
    """Convert between text-label and text-index."""

    def __init__(self,
                 gtc_label_decode=None,
                 character_dict_path=None,
                 use_space_char=True,
                 only_gtc=False,
                 **kwargs):
        gtc_label_decode['character_dict_path'] = character_dict_path
        gtc_label_decode['use_space_char'] = use_space_char
        self.gtc_label_decode = build_post_process(gtc_label_decode)
        self.ctc_label_decode = CTCLabelDecode(
            character_dict_path=character_dict_path,
            use_space_char=use_space_char)
        self.gtc_character = self.gtc_label_decode.character
        self.ctc_character = self.ctc_label_decode.character
        self.only_gtc = only_gtc

    def get_character_num(self):
        return [len(self.gtc_character), len(self.ctc_character)]

    def __call__(self, preds, batch=None, *args, **kwargs):
        gtc = self.gtc_label_decode(preds['gtc_pred'], batch[:-2])
        if self.only_gtc:
            return gtc
        ctc = self.ctc_label_decode(preds['ctc_pred'], [None] + batch[-2:])

        return [gtc, ctc]
