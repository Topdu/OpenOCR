import cv2
import six
import numpy as np

from .rec_aug import SVTRAug, ABINetAug
from .rec_aug import BaseDataAugmentation as BDA
from .resize import SVTRResize
from .ctc_label_encode import CTCLabelEncode, CPPDLabelEncode
from .char_label_encode import CharLabelEncode
from .ce_label_encode import CELabelEncode


class KeepKeys(object):
    def __init__(self, keep_keys, **kwargs):
        self.keep_keys = keep_keys

    def __call__(self, data):
        data_list = []
        for key in self.keep_keys:
            data_list.append(data[key])
        return data_list


def transform(data, ops=None):
    """transform"""
    if ops is None:
        ops = []
    for op in ops:
        data = op(data)
        if data is None:
            return None
    return data


class DecodeImage(object):
    """decode image"""

    def __init__(self,
                 img_mode="RGB",
                 channel_first=False,
                 ignore_orientation=False,
                 **kwargs):
        self.img_mode = img_mode
        self.channel_first = channel_first
        self.ignore_orientation = ignore_orientation

    def __call__(self, data):
        img = data["image"]
        if six.PY2:
            assert (type(img) is str and
                    len(img) > 0), "invalid input 'img' in DecodeImage"
        else:
            assert (type(img) is bytes and
                    len(img) > 0), "invalid input 'img' in DecodeImage"
        img = np.frombuffer(img, dtype="uint8")
        if self.ignore_orientation:
            img = cv2.imdecode(img, cv2.IMREAD_IGNORE_ORIENTATION |
                               cv2.IMREAD_COLOR)
        else:
            img = cv2.imdecode(img, 1)
        if img is None:
            return None
        if self.img_mode == "GRAY":
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif self.img_mode == "RGB":
            assert img.shape[2] == 3, "invalid shape of image[%s]" % (img.shape)
            img = img[:, :, ::-1]

        if self.channel_first:
            img = img.transpose((2, 0, 1))

        data["image"] = img
        return data


def create_operators(op_param_list, global_config=None):
    """
    create operators based on the config

    Args:
        params(list): a dict list, used to create some operators
    """
    assert isinstance(op_param_list, list), "operator config should be a list"
    ops = []
    for operator in op_param_list:
        assert isinstance(operator,
                          dict) and len(operator) == 1, "yaml format error"
        op_name = list(operator)[0]
        param = {} if operator[op_name] is None else operator[op_name]
        if global_config is not None:
            param.update(global_config)
        op = eval(op_name)(**param)
        ops.append(op)
    return ops
