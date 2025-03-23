import io
import copy
import importlib

import cv2
import numpy as np
from PIL import Image


class KeepKeys:

    def __init__(self, keep_keys, **kwargs):
        self.keep_keys = keep_keys

    def __call__(self, data):
        return [data[key] for key in self.keep_keys]


class Fasttext:

    def __init__(self, path='None', **kwargs):
        import fasttext
        self.fast_model = fasttext.load_model(path)

    def __call__(self, data):
        data['fast_label'] = self.fast_model[data['label']]
        return data


class DecodeImage:

    def __init__(self,
                 img_mode='RGB',
                 channel_first=False,
                 ignore_orientation=False,
                 **kwargs):
        self.img_mode = img_mode
        self.channel_first = channel_first
        self.ignore_orientation = ignore_orientation

    def __call__(self, data):
        assert isinstance(data['image'], bytes) and len(data['image']) > 0
        img = np.frombuffer(data['image'], dtype='uint8')

        flags = cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR if self.ignore_orientation else 1
        img = cv2.imdecode(img, flags)

        if self.img_mode == 'GRAY':
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif self.img_mode == 'RGB':
            img = img[:, :, ::-1]

        if self.channel_first:
            img = img.transpose((2, 0, 1))

        data['image'] = img
        return data


class DecodeImagePIL:

    def __init__(self, img_mode='RGB', **kwargs):
        self.img_mode = img_mode

    def __call__(self, data):
        assert isinstance(data['image'], bytes) and len(data['image']) > 0
        img = Image.open(io.BytesIO(data['image'])).convert('RGB')

        if self.img_mode == 'Gray':
            img = img.convert('L')
        elif self.img_mode == 'BGR':
            img = Image.fromarray(np.array(img)[:, :, ::-1])

        data['image'] = img
        return data


def transform(data, ops=None):
    """transform."""
    if ops is None:
        ops = []
    for op in ops:
        data = op(data)
        if data is None:
            return None
    return data


# 类名到模块的映射
MODULE_MAPPING = {
    'ABINetLabelEncode': '.abinet_label_encode',
    'ARLabelEncode': '.ar_label_encode',
    'CELabelEncode': '.ce_label_encode',
    'CharLabelEncode': '.char_label_encode',
    'CPPDLabelEncode': '.cppd_label_encode',
    'CTCLabelEncode': '.ctc_label_encode',
    'EPLabelEncode': '.ep_label_encode',
    'IGTRLabelEncode': '.igtr_label_encode',
    'MGPLabelEncode': '.mgp_label_encode',
    'SMTRLabelEncode': '.smtr_label_encode',
    'SRNLabelEncode': '.srn_label_encode',
    'VisionLANLabelEncode': '.visionlan_label_encode',
    'CAMLabelEncode': '.cam_label_encode',
    'ABINetAug': '.rec_aug',
    'BDA': '.rec_aug',
    'PARSeqAug': '.rec_aug',
    'PARSeqAugPIL': '.rec_aug',
    'SVTRAug': '.rec_aug',
    'ABINetResize': '.resize',
    'CDistNetResize': '.resize',
    'LongResize': '.resize',
    'RecTVResize': '.resize',
    'RobustScannerRecResizeImg': '.resize',
    'SliceResize': '.resize',
    'SliceTVResize': '.resize',
    'SRNRecResizeImg': '.resize',
    'SVTRResize': '.resize',
    'VisionLANResize': '.resize',
    'RecDynamicResize': '.resize',
}


def dynamic_import(class_name):
    module_path = MODULE_MAPPING.get(class_name)
    if not module_path:
        raise ValueError(f'Unsupported class: {class_name}')

    module = importlib.import_module(module_path, package=__package__)
    return getattr(module, class_name)


def create_operators(op_param_list, global_config=None):
    ops = []
    for op_info in op_param_list:
        op_name = list(op_info.keys())[0]
        param = copy.deepcopy(op_info[op_name]) or {}

        if global_config:
            param.update(global_config)

        if op_name in globals():
            op_class = globals()[op_name]
        else:
            op_class = dynamic_import(op_name)

        ops.append(op_class(**param))
    return ops


class GTCLabelEncode():
    """Convert between text-label and text-index."""

    def __init__(self,
                 gtc_label_encode,
                 max_text_length,
                 character_dict_path=None,
                 use_space_char=False,
                 **kwargs):
        self.gtc_label_encode = dynamic_import(gtc_label_encode['name'])(
            max_text_length=max_text_length,
            character_dict_path=character_dict_path,
            use_space_char=use_space_char,
            **gtc_label_encode)
        self.ctc_label_encode = dynamic_import('CTCLabelEncode')(
            max_text_length, character_dict_path, use_space_char)

    def __call__(self, data):
        data_ctc = self.ctc_label_encode({'label': data['label']})
        data = self.gtc_label_encode(data)
        if data_ctc is None or data is None:
            return None
        data['ctc_label'] = data_ctc['label']
        data['ctc_length'] = data_ctc['length']
        return data
