import copy

from .ctc_loss import CTCLoss

# from .char_loss import CharLoss
from .ce_loss import CELoss
from .cppd_loss import CPPDLoss

support_dict = ["CTCLoss", "CharLoss", "CELoss", "CPPDLoss"]


def build_loss(config):
    config = copy.deepcopy(config)
    module_name = config.pop("name")
    assert module_name in support_dict, Exception("loss only support {}".format(
        support_dict))
    module_class = eval(module_name)(**config)
    return module_class
