import copy

__all__ = ["build_post_process"]

from .ctc_postprocess import CTCLabelDecode
from .cppd_postprocess import CPPDLabelDecode
from .nrtr_postprocess import NRTRLabelDecode
from .char_postprocess import CharLabelDecode
from .ce_postprocess import CELabelDecode

support_dict = [
    "CTCLabelDecode",
    "CharLabelDecode",
    "CELabelDecode",
    "CPPDLabelDecode",
    "NRTRLabelDecode",
]


def build_post_process(config, global_config=None):
    config = copy.deepcopy(config)
    module_name = config.pop("name")
    if global_config is not None:
        config.update(global_config)
    assert module_name in support_dict, Exception(
        "post process only support {}".format(support_dict))
    module_class = eval(module_name)(**config)
    return module_class
