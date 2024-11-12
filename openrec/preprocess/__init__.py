import io

import cv2
import numpy as np
from PIL import Image

from .abinet_label_encode import ABINetLabelEncode
from .ar_label_encode import ARLabelEncode
from .ce_label_encode import CELabelEncode
from .char_label_encode import CharLabelEncode
from .cppd_label_encode import CPPDLabelEncode
from .ctc_label_encode import CTCLabelEncode
from .ep_label_encode import EPLabelEncode
from .igtr_label_encode import IGTRLabelEncode
from .mgp_label_encode import MGPLabelEncode
from .rec_aug import ABINetAug
from .rec_aug import BaseDataAugmentation as BDA
from .rec_aug import PARSeqAug, PARSeqAugPIL, SVTRAug
from .resize import (ABINetResize, CDistNetResize, LongResize, RecTVResize,
                     RobustScannerRecResizeImg, SliceResize, SliceTVResize,
                     SRNRecResizeImg, SVTRResize, VisionLANResize,
                     RecDynamicResize)
from .smtr_label_encode import SMTRLabelEncode
from .srn_label_encode import SRNLabelEncode
from .visionlan_label_encode import VisionLANLabelEncode
from .cam_label_encode import CAMLabelEncode


class KeepKeys(object):

    def __init__(self, keep_keys, **kwargs):
        self.keep_keys = keep_keys

    def __call__(self, data):
        data_list = []
        for key in self.keep_keys:
            data_list.append(data[key])
        return data_list


def transform(data, ops=None):
    """transform."""
    if ops is None:
        ops = []
    for op in ops:
        data = op(data)
        if data is None:
            return None
    return data


class Fasttext(object):

    def __init__(self, path='None', **kwargs):
        # pip install fasttext==0.9.1
        import fasttext

        self.fast_model = fasttext.load_model(path)

    def __call__(self, data):
        label = data['label']
        fast_label = self.fast_model[label]
        data['fast_label'] = fast_label
        return data


class DecodeImage(object):
    """decode image."""

    def __init__(self,
                 img_mode='RGB',
                 channel_first=False,
                 ignore_orientation=False,
                 **kwargs):
        self.img_mode = img_mode
        self.channel_first = channel_first
        self.ignore_orientation = ignore_orientation

    def __call__(self, data):
        img = data['image']

        assert type(img) is bytes and len(
            img) > 0, "invalid input 'img' in DecodeImage"
        img = np.frombuffer(img, dtype='uint8')
        if self.ignore_orientation:
            img = cv2.imdecode(
                img, cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR)
        else:
            img = cv2.imdecode(img, 1)
        if img is None:
            return None
        if self.img_mode == 'GRAY':
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif self.img_mode == 'RGB':
            assert img.shape[2] == 3, 'invalid shape of image[%s]' % (
                img.shape)
            img = img[:, :, ::-1]

        if self.channel_first:
            img = img.transpose((2, 0, 1))

        data['image'] = img
        return data


class DecodeImagePIL(object):
    """decode image."""

    def __init__(self, img_mode='RGB', **kwargs):
        self.img_mode = img_mode

    def __call__(self, data):
        img = data['image']
        assert type(img) is bytes and len(
            img) > 0, "invalid input 'img' in DecodeImage"
        img = data['image']
        buf = io.BytesIO(img)
        img = Image.open(buf).convert('RGB')
        if self.img_mode == 'Gray':
            img = img.convert('L')
        elif self.img_mode == 'BGR':
            img = np.array(img)[:, :, ::-1]  # 将图片转为numpy格式，并将最后一维通道倒序
            img = Image.fromarray(np.uint8(img))
        data['image'] = img
        return data


def create_operators(op_param_list, global_config=None):
    """create operators based on the config.

    Args:
        params(list): a dict list, used to create some operators
    """
    assert isinstance(op_param_list, list), 'operator config should be a list'
    ops = []
    for operator in op_param_list:
        assert isinstance(operator,
                          dict) and len(operator) == 1, 'yaml format error'
        op_name = list(operator)[0]
        param = {} if operator[op_name] is None else operator[op_name]
        if global_config is not None:
            param.update(global_config)
        op = eval(op_name)(**param)
        ops.append(op)
    return ops


class GTCLabelEncode():
    """Convert between text-label and text-index."""

    def __init__(self,
                 gtc_label_encode,
                 max_text_length,
                 character_dict_path=None,
                 use_space_char=False,
                 **kwargs):
        self.gtc_label_encode = eval(gtc_label_encode['name'])(
            max_text_length=max_text_length,
            character_dict_path=character_dict_path,
            use_space_char=use_space_char,
            **gtc_label_encode)
        self.ctc_label_encode = CTCLabelEncode(max_text_length,
                                               character_dict_path,
                                               use_space_char)

    def __call__(self, data):
        data_ctc = self.ctc_label_encode({'label': data['label']})
        data = self.gtc_label_encode(data)
        if data_ctc is None or data is None:
            return None
        data['ctc_label'] = data_ctc['label']
        data['ctc_length'] = data_ctc['length']
        return data
