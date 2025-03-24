import copy
import io
import cv2
import numpy as np
from PIL import Image
from importlib import import_module

MODULE_MAPPING = {
    'DetResizeForTest': '.db_resize_for_test',
    'CopyPaste': '.crop_paste',
    'IaaAugment': '.iaa_augment',
    'EastRandomCropData': '.crop_resize',
    'DetLabelEncode': '.db_label_encode',
    'MakeBorderMap': '.db_label_encode',
    'MakeShrinkMap': '.db_label_encode',
}


class NormalizeImage(object):
    """normalize image such as substract mean, divide std"""

    def __init__(self, scale=None, mean=None, std=None, order='chw', **kwargs):
        if isinstance(scale, str):
            scale = eval(scale)
        self.scale = np.float32(scale if scale is not None else 1.0 / 255.0)
        mean = mean if mean is not None else [0.485, 0.456, 0.406]
        std = std if std is not None else [0.229, 0.224, 0.225]

        shape = (3, 1, 1) if order == 'chw' else (1, 1, 3)
        self.mean = np.array(mean).reshape(shape).astype('float32')
        self.std = np.array(std).reshape(shape).astype('float32')

    def __call__(self, data):
        img = data['image']
        from PIL import Image

        if isinstance(img, Image.Image):
            img = np.array(img)
        assert isinstance(img,
                          np.ndarray), "invalid input 'img' in NormalizeImage"
        data['image'] = (img.astype('float32') * self.scale -
                         self.mean) / self.std
        return data


class ToCHWImage(object):
    """convert hwc image to chw image"""

    def __init__(self, **kwargs):
        pass

    def __call__(self, data):
        img = data['image']
        from PIL import Image

        if isinstance(img, Image.Image):
            img = np.array(img)
        data['image'] = img.transpose((2, 0, 1))
        return data


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


def dynamic_import(class_name):
    module_path = MODULE_MAPPING.get(class_name)
    if not module_path:
        raise ValueError(f'Unsupported class: {class_name}')

    module = import_module(module_path, package=__package__)
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
