import random

import cv2
import numpy as np
from PIL import Image
from .parseq_aug import rand_augment_transform


class PARSeqAugPIL(object):

    def __init__(self, **kwargs):
        self.transforms = rand_augment_transform()

    def __call__(self, data):
        img = data['image']
        img_aug = self.transforms(img)
        data['image'] = img_aug
        return data


class PARSeqAug(object):

    def __init__(self, **kwargs):
        self.transforms = rand_augment_transform()

    def __call__(self, data):
        img = data['image']

        img = np.array(self.transforms(Image.fromarray(img)))
        data['image'] = img
        return data


class ABINetAug(object):

    def __init__(self,
                 geometry_p=0.5,
                 deterioration_p=0.25,
                 colorjitter_p=0.25,
                 **kwargs):
        from torchvision.transforms import Compose
        from .abinet_aug import CVColorJitter, CVDeterioration, CVGeometry
        self.transforms = Compose([
            CVGeometry(
                degrees=45,
                translate=(0.0, 0.0),
                scale=(0.5, 2.0),
                shear=(45, 15),
                distortion=0.5,
                p=geometry_p,
            ),
            CVDeterioration(var=20, degrees=6, factor=4, p=deterioration_p),
            CVColorJitter(
                brightness=0.5,
                contrast=0.5,
                saturation=0.5,
                hue=0.1,
                p=colorjitter_p,
            ),
        ])

    def __call__(self, data):
        img = data['image']
        img = self.transforms(img)
        data['image'] = img
        return data


class SVTRAug(object):

    def __init__(self,
                 aug_type=0,
                 geometry_p=0.5,
                 deterioration_p=0.25,
                 colorjitter_p=0.25,
                 **kwargs):
        from torchvision.transforms import Compose
        from .abinet_aug import CVColorJitter, SVTRDeterioration, SVTRGeometry
        self.transforms = Compose([
            SVTRGeometry(
                aug_type=aug_type,
                degrees=45,
                translate=(0.0, 0.0),
                scale=(0.5, 2.0),
                shear=(45, 15),
                distortion=0.5,
                p=geometry_p,
            ),
            SVTRDeterioration(var=20, degrees=6, factor=4, p=deterioration_p),
            CVColorJitter(
                brightness=0.5,
                contrast=0.5,
                saturation=0.5,
                hue=0.1,
                p=colorjitter_p,
            ),
        ])

    def __call__(self, data):
        img = data['image']
        img = self.transforms(img)
        data['image'] = img
        return data


class BaseDataAugmentation(object):

    def __init__(self,
                 crop_prob=0.4,
                 reverse_prob=0.4,
                 noise_prob=0.4,
                 jitter_prob=0.4,
                 blur_prob=0.4,
                 hsv_aug_prob=0.4,
                 **kwargs):
        self.crop_prob = crop_prob
        self.reverse_prob = reverse_prob
        self.noise_prob = noise_prob
        self.jitter_prob = jitter_prob
        self.blur_prob = blur_prob
        self.hsv_aug_prob = hsv_aug_prob
        # for GaussianBlur
        self.fil = cv2.getGaussianKernel(ksize=5, sigma=1, ktype=cv2.CV_32F)

    def __call__(self, data):
        img = data['image']
        h, w, _ = img.shape

        if random.random() <= self.crop_prob and h >= 20 and w >= 20:
            img = get_crop(img)

        if random.random() <= self.blur_prob:
            # GaussianBlur
            img = cv2.sepFilter2D(img, -1, self.fil, self.fil)

        if random.random() <= self.hsv_aug_prob:
            img = hsv_aug(img)

        if random.random() <= self.jitter_prob:
            img = jitter(img)

        if random.random() <= self.noise_prob:
            img = add_gasuss_noise(img)

        if random.random() <= self.reverse_prob:
            img = 255 - img

        data['image'] = img
        return data


def hsv_aug(img):
    """cvtColor."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    delta = 0.001 * random.random() * flag()
    hsv[:, :, 2] = hsv[:, :, 2] * (1 + delta)
    new_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return new_img


def blur(img):
    """blur."""
    h, w, _ = img.shape
    if h > 10 and w > 10:
        return cv2.GaussianBlur(img, (5, 5), 1)
    else:
        return img


def jitter(img):
    """jitter."""
    w, h, _ = img.shape
    if h > 10 and w > 10:
        thres = min(w, h)
        s = int(random.random() * thres * 0.01)
        src_img = img.copy()
        for i in range(s):
            img[i:, i:, :] = src_img[:w - i, :h - i, :]
        return img
    else:
        return img


def add_gasuss_noise(image, mean=0, var=0.1):
    """Gasuss noise."""

    noise = np.random.normal(mean, var**0.5, image.shape)
    out = image + 0.5 * noise
    out = np.clip(out, 0, 255)
    out = np.uint8(out)
    return out


def get_crop(image):
    """random crop."""
    h, w, _ = image.shape
    top_min = 1
    top_max = 8
    top_crop = int(random.randint(top_min, top_max))
    top_crop = min(top_crop, h - 1)
    crop_img = image.copy()
    ratio = random.randint(0, 1)
    if ratio:
        crop_img = crop_img[top_crop:h, :, :]
    else:
        crop_img = crop_img[0:h - top_crop, :, :]
    return crop_img


def flag():
    """flag."""
    return 1 if random.random() > 0.5000001 else -1
