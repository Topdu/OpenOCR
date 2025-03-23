import math
import random

import cv2
import numpy as np
from PIL import Image


class CDistNetResize(object):

    def __init__(self, image_shape, **kwargs):
        self.image_shape = image_shape

    def __call__(self, data):
        img = data['image']
        _, h, w = self.image_shape
        # keep_aspect_ratio = False
        image_pil = Image.fromarray(np.uint8(img))
        image = image_pil.resize((w, h), Image.LANCZOS)
        image = np.array(image)
        # rgb2gray = False
        image = image.transpose((2, 0, 1))
        image = image.astype(np.float32) / 128.0 - 1.0
        data['image'] = image
        data['valid_ratio'] = 1
        return data


class ABINetResize(object):

    def __init__(self, image_shape, **kwargs):
        self.image_shape = image_shape

    def __call__(self, data):
        img = data['image']
        h, w = img.shape[:2]
        norm_img, valid_ratio = resize_norm_img_abinet(img, self.image_shape)
        data['image'] = norm_img
        data['valid_ratio'] = valid_ratio
        r = float(w) / float(h)
        data['real_ratio'] = max(1, round(r))
        return data


def resize_norm_img_abinet(img, image_shape):
    imgC, imgH, imgW = image_shape

    resized_image = cv2.resize(img, (imgW, imgH),
                               interpolation=cv2.INTER_LINEAR)
    resized_w = imgW
    resized_image = resized_image.astype('float32')
    resized_image = resized_image / 255.0

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    resized_image = (resized_image - mean[None, None, ...]) / std[None, None,
                                                                  ...]
    resized_image = resized_image.transpose((2, 0, 1))
    resized_image = resized_image.astype('float32')

    valid_ratio = min(1.0, float(resized_w / imgW))
    return resized_image, valid_ratio


class SVTRResize(object):

    def __init__(self, image_shape, padding=True, **kwargs):
        self.image_shape = image_shape
        self.padding = padding

    def __call__(self, data):
        img = data['image']
        h, w = img.shape[:2]
        norm_img, valid_ratio = resize_norm_img(img, self.image_shape,
                                                self.padding)
        data['image'] = norm_img
        data['valid_ratio'] = valid_ratio
        r = float(w) / float(h)
        data['real_ratio'] = max(1, round(r))
        return data


class RecTVResize(object):

    def __init__(self, image_shape=[32, 128], padding=True, **kwargs):
        from torchvision import transforms as T
        from torchvision.transforms import functional as F
        self.F = F
        self.padding = padding
        self.image_shape = image_shape
        self.interpolation = T.InterpolationMode.BICUBIC
        transforms = []
        transforms.extend([
            T.ToTensor(),
            T.Normalize(0.5, 0.5),
        ])
        self.transforms = T.Compose(transforms)

    def __call__(self, data):
        img = data['image']
        imgH, imgW = self.image_shape
        w, h = img.size
        if not self.padding:
            resized_w = imgW
        else:
            ratio = w / float(h)
            if math.ceil(imgH * ratio) > imgW:
                resized_w = imgW
            else:
                resized_w = int(math.ceil(imgH * ratio))
        resized_image = self.F.resize(img, (imgH, resized_w),
                                      interpolation=self.interpolation)
        img = self.transforms(resized_image)
        if resized_w < imgW:
            img = self.F.pad(img, [0, 0, imgW - resized_w, 0], fill=0.)
        valid_ratio = min(1.0, float(resized_w / imgW))
        data['image'] = img
        data['valid_ratio'] = valid_ratio
        r = float(w) / float(h)
        data['real_ratio'] = max(1, round(r))
        return data


class LongResize(object):

    def __init__(self,
                 base_shape=[[64, 64], [96, 48], [112, 40], [128, 32]],
                 max_ratio=12,
                 base_h=32,
                 padding_rand=False,
                 padding_bi=False,
                 padding=True,
                 **kwargs):
        self.base_shape = base_shape
        self.max_ratio = max_ratio
        self.base_h = base_h
        self.padding = padding
        self.padding_rand = padding_rand
        self.padding_bi = padding_bi

    def __call__(self, data):
        data = resize_norm_img_long(
            data,
            self.base_shape,
            self.max_ratio,
            self.base_h,
            self.padding,
            self.padding_rand,
            self.padding_bi,
        )
        return data


class SliceResize(object):

    def __init__(self, image_shape, padding=True, max_ratio=12, **kwargs):
        self.image_shape = image_shape
        self.padding = padding
        self.max_ratio = max_ratio

    def __call__(self, data):
        img = data['image']
        h, w = img.shape[:2]
        w_bi = w // 2
        img_list = [
            img[:, :w_bi, :], img[:, w_bi:2 * w_bi, :],
            img[:, w_bi // 2:(w_bi // 2) + w_bi, :]
        ]
        img_reshape = []
        for img_s in img_list:
            norm_img, valid_ratio = resize_norm_img_slice(
                img_s, self.image_shape, max_ratio=self.max_ratio)
            img_reshape.append(norm_img[None, :, :, :])
        data['image'] = np.concatenate(img_reshape, 0)
        data['valid_ratio'] = valid_ratio
        return data


class SliceTVResize(object):

    def __init__(self,
                 image_shape,
                 padding=True,
                 base_shape=[[64, 64], [96, 48], [112, 40], [128, 32]],
                 max_ratio=12,
                 base_h=32,
                 **kwargs):
        import torch
        from torchvision import transforms as T
        from torchvision.transforms import functional as F
        self.F = F
        self.torch = torch
        self.image_shape = image_shape
        self.padding = padding
        self.max_ratio = max_ratio
        self.base_h = base_h
        self.interpolation = T.InterpolationMode.BICUBIC
        transforms = []
        transforms.extend([
            T.ToTensor(),
            T.Normalize(0.5, 0.5),
        ])
        self.transforms = T.Compose(transforms)

    def __call__(self, data):
        img = data['image']
        w, h = img.size
        w_ratio = ((w // h) // 2) * 2
        w_ratio = max(6, w_ratio)
        img = self.F.resize(img, (self.base_h, self.base_h * w_ratio),
                            interpolation=self.interpolation)
        img = self.transforms(img)
        img_list = []
        for i in range(0, w_ratio // 2 - 1):
            img_list.append(img[None, :, :,
                                i * 2 * self.base_h:(i * 2 + 4) * self.base_h])
        data['image'] = self.torch.concat(img_list, 0)
        data['valid_ratio'] = float(w_ratio) / w
        return data


class RecTVResizeRatio(object):

    def __init__(self,
                 image_shape=[32, 128],
                 padding=True,
                 base_shape=[[64, 64], [96, 48], [112, 40], [128, 32]],
                 max_ratio=12,
                 base_h=32,
                 **kwargs):
        from torchvision import transforms as T
        from torchvision.transforms import functional as F
        self.F = F
        self.padding = padding
        self.image_shape = image_shape
        self.max_ratio = max_ratio
        self.base_shape = base_shape
        self.base_h = base_h
        self.interpolation = T.InterpolationMode.BICUBIC
        transforms = []
        transforms.extend([
            T.ToTensor(),
            T.Normalize(0.5, 0.5),
        ])
        self.transforms = T.Compose(transforms)

    def __call__(self, data):
        img = data['image']
        imgH, imgW = self.image_shape
        w, h = img.size
        gen_ratio = round(float(w) / float(h))
        ratio_resize = 1 if gen_ratio == 0 else gen_ratio
        ratio_resize = min(ratio_resize, self.max_ratio)
        imgW, imgH = self.base_shape[ratio_resize -
                                     1] if ratio_resize <= 4 else [
                                         self.base_h *
                                         ratio_resize, self.base_h
                                     ]
        if not self.padding:
            resized_w = imgW
        else:
            ratio = w / float(h)
            if math.ceil(imgH * ratio) > imgW:
                resized_w = imgW
            else:
                resized_w = int(math.ceil(imgH * ratio))
        resized_image = self.F.resize(img, (imgH, resized_w),
                                      interpolation=self.interpolation)
        img = self.transforms(resized_image)
        if resized_w < imgW:
            img = self.F.pad(img, [0, 0, imgW - resized_w, 0], fill=0.)
        valid_ratio = min(1.0, float(resized_w / imgW))
        data['image'] = img
        data['valid_ratio'] = valid_ratio
        return data


class RecDynamicResize(object):

    def __init__(self, image_shape=[32, 128], padding=True, **kwargs):
        self.padding = padding
        self.image_shape = image_shape
        self.max_ratio = image_shape[1] * 1.0 / image_shape[0]

    def __call__(self, data):
        img = data['image']
        imgH, imgW = self.image_shape
        h, w, imgC = img.shape
        ratio = w / float(h)
        max_wh_ratio = max(ratio, self.max_ratio)
        imgW = int(imgH * max_wh_ratio)
        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = int(math.ceil(imgH * ratio))
        resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resized_image.astype('float32')
        resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image
        data['image'] = padding_im
        return data


def resize_norm_img_slice(
    img,
    image_shape,
    base_shape=[[64, 64], [96, 48], [112, 40], [128, 32]],
    max_ratio=12,
    base_h=32,
    padding=True,
):
    imgC, imgH, imgW = image_shape
    h = img.shape[0]
    w = img.shape[1]
    gen_ratio = round(float(w) / float(h))
    ratio_resize = 1 if gen_ratio == 0 else gen_ratio
    ratio_resize = min(ratio_resize, max_ratio)
    imgW, imgH = base_shape[ratio_resize - 1] if ratio_resize <= 4 else [
        base_h * ratio_resize, base_h
    ]
    padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
    if not padding:
        resized_image = cv2.resize(img, (imgW, imgH))
        resized_w = imgW
    else:
        ratio = w / float(h)
        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = int(math.ceil(imgH * ratio * (random.random() + 0.5)))
            resized_w = min(imgW, resized_w)
        resized_image = cv2.resize(img, (resized_w, imgH))
    resized_image = resized_image.transpose((2, 0, 1)) / 255
    resized_image -= 0.5
    resized_image /= 0.5
    padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
    padding_im[:, :, :resized_w] = resized_image
    valid_ratio = min(1.0, float(resized_w / imgW))
    return padding_im, valid_ratio


def resize_norm_img(img,
                    image_shape,
                    padding=True,
                    interpolation=cv2.INTER_LINEAR):
    imgC, imgH, imgW = image_shape
    h = img.shape[0]
    w = img.shape[1]
    if not padding:
        resized_image = cv2.resize(img, (imgW, imgH),
                                   interpolation=interpolation)
        resized_w = imgW
    else:
        ratio = w / float(h)
        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = int(math.ceil(imgH * ratio))
        resized_image = cv2.resize(img, (resized_w, imgH))
    resized_image = resized_image.astype('float32')
    if image_shape[0] == 1:
        resized_image = resized_image / 255
        resized_image = resized_image[np.newaxis, :]
    else:
        resized_image = resized_image.transpose((2, 0, 1)) / 255
    resized_image -= 0.5
    resized_image /= 0.5
    padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
    padding_im[:, :, 0:resized_w] = resized_image
    valid_ratio = min(1.0, float(resized_w / imgW))
    return padding_im, valid_ratio


def resize_norm_img_long(
    data,
    base_shape=[[64, 64], [96, 48], [112, 40], [128, 32]],
    max_ratio=12,
    base_h=32,
    padding=True,
    padding_rand=False,
    padding_bi=False,
):
    img = data['image']
    h = img.shape[0]
    w = img.shape[1]
    gen_ratio = data.get('gen_ratio', 0)
    if gen_ratio == 0:
        ratio = w / float(h)
        gen_ratio = round(ratio) if ratio > 0.5 else 1
    gen_ratio = min(data['gen_ratio'], max_ratio)
    if padding_rand and random.random() < 0.5:
        padding = False if padding else True
    imgW, imgH = base_shape[gen_ratio -
                            1] if gen_ratio <= len(base_shape) else [
                                base_h * gen_ratio, base_h
                            ]
    if not padding:
        resized_image = cv2.resize(img, (imgW, imgH),
                                   interpolation=cv2.INTER_LINEAR)
        resized_w = imgW
    else:
        ratio = w / float(h)
        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = int(math.ceil(imgH * ratio * (random.random() + 0.5)))
            resized_w = min(imgW, resized_w)

        resized_image = cv2.resize(img, (resized_w, imgH))
    resized_image = resized_image.astype('float32')

    resized_image = resized_image.transpose((2, 0, 1)) / 255
    resized_image -= 0.5
    resized_image /= 0.5
    padding_im = np.zeros((3, imgH, imgW), dtype=np.float32)
    if padding_bi and random.random() < 0.5:
        padding_im[:, :, -resized_w:] = resized_image
    else:
        padding_im[:, :, :resized_w] = resized_image
    valid_ratio = min(1.0, float(resized_w / imgW))
    data['image'] = padding_im
    data['valid_ratio'] = valid_ratio
    data['gen_ratio'] = imgW // imgH
    data['real_ratio'] = w // h
    return data


class VisionLANResize(object):

    def __init__(self, image_shape, **kwargs):
        self.image_shape = image_shape

    def __call__(self, data):
        img = data['image']

        imgC, imgH, imgW = self.image_shape
        resized_image = cv2.resize(img, (imgW, imgH))
        resized_image = resized_image.astype('float32')
        if imgC == 1:
            resized_image = resized_image / 255
            norm_img = resized_image[np.newaxis, :]
        else:
            norm_img = resized_image.transpose((2, 0, 1)) / 255

        data['image'] = norm_img
        data['valid_ratio'] = 1.0
        return data


class RobustScannerRecResizeImg(object):

    def __init__(self, image_shape, width_downsample_ratio=0.25, **kwargs):
        self.image_shape = image_shape
        self.width_downsample_ratio = width_downsample_ratio

    def __call__(self, data):
        img = data['image']
        norm_img, resize_shape, pad_shape, valid_ratio = resize_norm_img_sar(
            img, self.image_shape, self.width_downsample_ratio)
        data['image'] = norm_img
        data['resized_shape'] = resize_shape
        data['pad_shape'] = pad_shape
        data['valid_ratio'] = valid_ratio
        return data


def resize_norm_img_sar(img, image_shape, width_downsample_ratio=0.25):
    imgC, imgH, imgW_min, imgW_max = image_shape
    h = img.shape[0]
    w = img.shape[1]
    valid_ratio = 1.0
    # make sure new_width is an integral multiple of width_divisor.
    width_divisor = int(1 / width_downsample_ratio)
    # resize
    ratio = w / float(h)
    resize_w = math.ceil(imgH * ratio)
    if resize_w % width_divisor != 0:
        resize_w = round(resize_w / width_divisor) * width_divisor
    if imgW_min is not None:
        resize_w = max(imgW_min, resize_w)
    if imgW_max is not None:
        valid_ratio = min(1.0, 1.0 * resize_w / imgW_max)
        resize_w = min(imgW_max, resize_w)
    resized_image = cv2.resize(img, (resize_w, imgH))
    resized_image = resized_image.astype('float32')
    # norm
    if image_shape[0] == 1:
        resized_image = resized_image / 255
        resized_image = resized_image[np.newaxis, :]
    else:
        resized_image = resized_image.transpose((2, 0, 1)) / 255
    resized_image -= 0.5
    resized_image /= 0.5
    resize_shape = resized_image.shape
    padding_im = -1.0 * np.ones((imgC, imgH, imgW_max), dtype=np.float32)
    padding_im[:, :, 0:resize_w] = resized_image
    pad_shape = padding_im.shape

    return padding_im, resize_shape, pad_shape, valid_ratio


class SRNRecResizeImg(object):

    def __init__(self, image_shape, **kwargs):
        self.image_shape = image_shape

    def __call__(self, data):
        img = data['image']
        norm_img = resize_norm_img_srn(img, self.image_shape)
        data['image'] = norm_img

        return data


def resize_norm_img_srn(img, image_shape):
    imgC, imgH, imgW = image_shape

    img_black = np.zeros((imgH, imgW))
    im_hei = img.shape[0]
    im_wid = img.shape[1]

    if im_wid <= im_hei * 1:
        img_new = cv2.resize(img, (imgH * 1, imgH))
    elif im_wid <= im_hei * 2:
        img_new = cv2.resize(img, (imgH * 2, imgH))
    elif im_wid <= im_hei * 3:
        img_new = cv2.resize(img, (imgH * 3, imgH))
    else:
        img_new = cv2.resize(img, (imgW, imgH))

    img_np = np.asarray(img_new)
    img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    img_black[:, 0:img_np.shape[1]] = img_np
    img_black = img_black[:, :, np.newaxis]

    row, col, c = img_black.shape
    c = 1

    return np.reshape(img_black, (c, row, col)).astype(np.float32)
