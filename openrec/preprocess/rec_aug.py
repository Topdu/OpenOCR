import random

import cv2
import numpy as np
from PIL import Image, ImageOps


class PARSeqAugPIL(object):

    def __init__(self, **kwargs):
        from .parseq_aug import rand_augment_transform
        self.transforms = rand_augment_transform()

    def __call__(self, data):
        img = data['image']
        img_aug = self.transforms(img)
        data['image'] = img_aug
        return data


class PARSeqAug(object):

    def __init__(self, **kwargs):
        from .parseq_aug import rand_augment_transform
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


class DocAug(object):

    def __init__(self, **kwargs):
        import albumentations as A
        self.aug = A.Compose(
            [
                A.RandomShadow(
                    shadow_roi=(0, 0.7, 1, 1), num_shadows_upper=2, p=0.5),
                # 透视变换，模拟拍照角度，这个计算量稍大，但效果关键
                A.OneOf(
                    [
                        A.Perspective(
                            scale=(0.05, 0.1),  # 控制形变程度
                            keep_size=False,
                            fit_output=True,  # 保持输出大小一致
                            pad_mode=cv2.BORDER_CONSTANT,
                            pad_val=[255, 255, 255],  # 白色填充
                            p=0.3  # 50% 概率应用
                        ),
                        A.GridDistortion(distort_limit=0.1,
                                         border_mode=0,
                                         interpolation=3,
                                         value=[255, 255, 255],
                                         p=0.3),
                        # 安全版弹性变换
                        A.ElasticTransform(
                            alpha=0.5,  # 减小形变强度
                            sigma=30,  # 增大平滑度
                            border_mode=cv2.BORDER_REPLICATE,  # 使用边缘复制代替裁剪
                            p=0.3)
                    ],
                    p=0.3),
                A.RGBShift(r_shift_limit=15,
                           g_shift_limit=15,
                           b_shift_limit=15,
                           p=0.3),

                # 光学特性模拟
                A.OneOf(
                    [
                        A.MotionBlur(blur_limit=(3, 5), p=0.7),
                        A.GaussianBlur(blur_limit=(3, 5), p=0.7),
                        # A.GlassBlur(sigma=0.05, max_delta=1, iterations=1, p=0.2)
                    ],
                    p=0.6),

                # 色彩空间变换
                A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2),
                                           contrast_limit=(-0.1, 0.1),
                                           p=0.5),
                A.RandomGamma(gamma_limit=(80, 120), p=0.3),
                A.CLAHE(clip_limit=2.0, p=0.2),

                # 压缩伪影
                A.ImageCompression(quality_lower=5, quality_upper=90, p=0.8),

                # 传感器噪声模拟
                A.OneOf([
                    A.GaussNoise(var_limit=(10.0, 30.0), p=0.5),
                    A.ISONoise(
                        color_shift=(0.01, 0.05), intensity=(0.1, 0.3), p=0.3),
                    A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=0.2)
                ],
                        p=0.8),
                A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.5),

                # 随机分辨率调整（保持长宽比）
                A.Downscale(scale_min=0.5,
                            scale_max=0.8,
                            interpolation=cv2.INTER_LINEAR,
                            p=0.25),
                A.PixelDropout(dropout_prob=0.01, p=0.2)
            ],
            p=1)  # 100%的概率应用整个流程

    def biased_random_int(self, max_value: int, exponent: float = 3.0) -> int:
        """
        返回 [0, max_value] 之间的一个整数，值越大概率越小。
        exponent > 1 越大表示越偏向小值。
        """
        r = random.random()  # 均匀分布 [0,1)
        biased = r**exponent  # 变成偏小
        return int(biased * max_value)

    def random_pad(self, img: Image.Image) -> Image.Image:
        w, h = img.size

        # 最大可 pad 宽度和高度
        max_pad_w = int(w * 0.2)
        max_pad_h = int(h * 0.2)

        # 随机分配左右 padding
        pad_left = self.biased_random_int(max_pad_w)
        pad_right = self.biased_random_int(max_pad_w)

        # 随机分配上下 padding
        pad_top = self.biased_random_int(max_pad_h)
        pad_bottom = self.biased_random_int(max_pad_h)

        # 应用 padding
        padded_img = ImageOps.expand(img,
                                     border=(pad_left, pad_top, pad_right,
                                             pad_bottom),
                                     fill=(255, 255, 255))
        return padded_img

    def __call__(self, data):
        image = data['image']
        arxiv = data['arxiv']
        # 执行数据增强
        try:
            if random.random() < 0.2 and arxiv:
                # 随机填充
                image = self.random_pad(image)
            if not isinstance(image, np.ndarray):
                image = np.array(image)

            augmented = self.aug(image=image)['image']
            data['image'] = Image.fromarray(augmented)
        except Exception as e:
            print(e)

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
