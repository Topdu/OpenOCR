from __future__ import annotations
import math
from typing import List, Optional, Tuple, Union

import numpy as np
from scipy.ndimage import zoom as scizoom

# Transformers imports
from transformers.image_processing_utils import BaseImageProcessor, BatchFeature
from transformers.image_transforms import convert_to_rgb, resize, to_channel_dimension_format
from transformers.image_utils import (
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    infer_channel_dimension_format,
    make_flat_list_of_images,
    to_numpy_array,
    valid_images,
    validate_preprocess_arguments,
)
from transformers import PreTrainedTokenizerFast
from transformers.utils import TensorType, filter_out_non_signature_kwargs, is_vision_available, logging
from transformers import AutoImageProcessor, ProcessorMixin
import torch
# Third-party optional imports
logger = logging.get_logger(__name__)

try:
    import albumentations as A
except Exception as _e:
    A = None
    _A_IMPORT_ERR = str(_e)

try:
    import cv2
except Exception:
    cv2 = None

if is_vision_available():
    from PIL import Image, ImageOps, ImageDraw

# Albumentations Custom Transforms
if A is not None:

    class Erosion(A.ImageOnlyTransform):

        def __init__(self, scale, always_apply=False, p=0.5):
            super().__init__(always_apply=always_apply, p=p)
            if type(scale) is tuple or type(scale) is list:
                assert len(scale) == 2
                self.scale = scale
            else:
                self.scale = (scale, scale)

        def apply(self, img, **params):
            if cv2 is None:
                return img
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                tuple(np.random.randint(self.scale[0], self.scale[1], 2)))
            img = cv2.erode(img, kernel, iterations=1)
            return img

    class Dilation(A.ImageOnlyTransform):

        def __init__(self, scale, always_apply=False, p=0.5):
            super().__init__(always_apply=always_apply, p=p)
            if type(scale) is tuple or type(scale) is list:
                assert len(scale) == 2
                self.scale = scale
            else:
                self.scale = (scale, scale)

        def apply(self, img, **params):
            if cv2 is None:
                return img
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                tuple(np.random.randint(self.scale[0], self.scale[1], 2)))
            img = cv2.dilate(img, kernel, iterations=1)
            return img

    class Bitmap(A.ImageOnlyTransform):

        def __init__(self, value=0, lower=200, p=0.5):
            super().__init__(p=p)
            self.lower = lower
            self.value = value

        def apply(self, img, **params):
            img = img.copy()
            img[img < self.lower] = self.value
            return img

    class Fog(A.ImageOnlyTransform):

        def __init__(self, mag=-1, always_apply=False, p=1.):
            super().__init__(always_apply=always_apply, p=p)
            self.rng = np.random.default_rng()
            self.mag = mag

        def apply(self, img, **params):
            img = Image.fromarray(img.astype(np.uint8))
            w, h = img.size
            c = [(1.5, 2), (2., 2), (2.5, 1.7)]
            if self.mag < 0 or self.mag >= len(c):
                index = self.rng.integers(0, len(c))
            else:
                index = self.mag
            c = c[index]
            n_channels = len(img.getbands())
            isgray = n_channels == 1
            img = np.asarray(img) / 255.
            max_val = img.max()
            max_size = 2**math.ceil(math.log2(max(w, h)) + 1)
            fog = c[0] * plasma_fractal(mapsize=max_size,
                                        wibbledecay=c[1],
                                        rng=self.rng)[:h, :w][..., np.newaxis]
            if isgray:
                fog = np.squeeze(fog)
            else:
                fog = np.repeat(fog, 3, axis=2)
            img += fog
            img = np.clip(img * max_val / (max_val + c[0]), 0, 1) * 255
            return img.astype(np.uint8)

    class Frost(A.ImageOnlyTransform):

        def __init__(self, mag=-1, always_apply=False, p=1.):
            super().__init__(always_apply=always_apply, p=p)
            self.rng = np.random.default_rng()
            self.mag = mag

        def apply(self, img, **params):
            img = Image.fromarray(img.astype(np.uint8))
            w, h = img.size
            c = [(0.78, 0.22), (0.64, 0.36), (0.5, 0.5)]
            if self.mag < 0 or self.mag >= len(c):
                index = self.rng.integers(0, len(c))
            else:
                index = self.mag
            c = c[index]
            filename = [
                './openrec/preprocess/cmer_frost/frost1.png',
                './openrec/preprocess/cmer_frost/frost2.png',
                './openrec/preprocess/cmer_frost/frost3.png',
                './openrec/preprocess/cmer_frost/frost4.jpg',
                './openrec/preprocess/cmer_frost/frost5.jpg',
                './openrec/preprocess/cmer_frost/frost6.jpg',
            ]
            index = self.rng.integers(0, len(filename))
            filename = filename[index]
            try:
                frost = Image.open(filename).convert('RGB')
            except Exception:
                # Fallback if file not found
                return np.asarray(img).astype(np.uint8)

            f_w, f_h = frost.size
            if w / h > f_w / f_h:
                f_h = round(f_h * w / f_w)
                f_w = w
            else:
                f_w = round(f_w * h / f_h)
                f_h = h
            frost = np.asarray(frost.resize((f_w, f_h)))
            y_start = self.rng.integers(0, f_h - h + 1)
            x_start = self.rng.integers(0, f_w - w + 1)
            frost = frost[y_start:y_start + h, x_start:x_start + w]
            n_channels = len(img.getbands())
            isgray = n_channels == 1
            img = np.asarray(img)
            if isgray:
                img = np.expand_dims(img, axis=2)
                img = np.repeat(img, 3, axis=2)
            img = np.clip(np.round(c[0] * img + c[1] * frost), 0, 255)
            img = img.astype(np.uint8)
            if isgray:
                img = np.squeeze(img)
            return img

    class Snow(A.ImageOnlyTransform):

        def __init__(self, mag=-1, always_apply=False, p=1.):
            super().__init__(always_apply=always_apply, p=p)
            self.rng = np.random.default_rng()
            self.mag = mag

        def apply(self, img, **params):
            img_pil = Image.fromarray(img.astype(np.uint8))
            w, h = img_pil.size
            c = [(0.1, 0.3, 3, 0.5, 10, 4, 0.8),
                 (0.2, 0.3, 2, 0.5, 12, 4, 0.7),
                 (0.55, 0.3, 4, 0.9, 12, 8, 0.7)]
            if self.mag < 0 or self.mag >= len(c):
                index = self.rng.integers(0, len(c))
            else:
                index = self.mag
            c = c[index]
            isgray = (len(img_pil.getbands()) == 1)
            img = np.asarray(img_pil, dtype=np.float32) / 255.
            if isgray:
                img = np.repeat(img[..., None], 3, axis=2)
            snow_layer = self.rng.normal(loc=c[0],
                                         scale=c[1],
                                         size=img.shape[:2])
            snow_layer[snow_layer < c[3]] = 0
            snow_layer = np.clip(snow_layer, 0, 1).astype(np.float32)
            angle = float(self.rng.uniform(-135, -45))
            snow_layer = motion_blur(snow_layer,
                                     radius=c[4],
                                     sigma=c[5],
                                     angle=angle)
            snow_layer = snow_layer[..., None]
            img = c[6] * img
            if cv2 is not None:
                gray_img = (1 - c[6]) * np.maximum(
                    img,
                    cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).reshape(h, w, 1) *
                    1.5 + 0.5)
                img += gray_img
            img = np.clip(img + snow_layer + np.rot90(snow_layer, k=2), 0,
                          1) * 255
            img = img.astype(np.uint8)
            return np.squeeze(img) if isgray else img

    class Rain(A.ImageOnlyTransform):

        def __init__(self, mag=-1, always_apply=False, p=1.):
            super().__init__(always_apply=always_apply, p=p)
            self.rng = np.random.default_rng()
            self.mag = mag

        def apply(self, img, **params):
            img = Image.fromarray(img.astype(np.uint8))
            img = img.copy()
            w, h = img.size
            n_channels = len(img.getbands())
            isgray = n_channels == 1
            line_width = self.rng.integers(1, 2)
            c = [50, 70, 90]
            if self.mag < 0 or self.mag >= len(c):
                index = 0
            else:
                index = self.mag
            c = c[index]
            n_rains = self.rng.integers(c, c + 20)
            slant = self.rng.integers(-60, 60)
            fillcolor = 200 if isgray else (200, 200, 200)
            draw = ImageDraw.Draw(img)
            max_length = min(w, h, 10)
            for i in range(1, n_rains):
                length = self.rng.integers(5, max_length)
                x1 = self.rng.integers(0, w - length)
                y1 = self.rng.integers(0, h - length)
                x2 = x1 + length * math.sin(slant * math.pi / 180.)
                y2 = y1 + length * math.cos(slant * math.pi / 180.)
                x2 = int(x2)
                y2 = int(y2)
                draw.line([(x1, y1), (x2, y2)],
                          width=line_width,
                          fill=fillcolor)
            img = np.asarray(img).astype(np.uint8)
            return img

    class Shadow(A.ImageOnlyTransform):

        def __init__(self, mag=-1, always_apply=False, p=1.):
            super().__init__(always_apply=always_apply, p=p)
            self.rng = np.random.default_rng()
            self.mag = mag

        def apply(self, img, **params):
            img = Image.fromarray(img.astype(np.uint8))
            w, h = img.size
            n_channels = len(img.getbands())
            isgray = n_channels == 1
            c = [64, 96, 128]
            if self.mag < 0 or self.mag >= len(c):
                index = 0
            else:
                index = self.mag
            c = c[index]
            img = img.convert('RGBA')
            overlay = Image.new('RGBA', img.size, (255, 255, 255, 0))
            draw = ImageDraw.Draw(overlay)
            transparency = self.rng.integers(c, c + 32)
            x1 = self.rng.integers(0, w // 2)
            y1 = 0
            x2 = self.rng.integers(w // 2, w)
            y2 = 0
            x3 = self.rng.integers(w // 2, w)
            y3 = h - 1
            x4 = self.rng.integers(0, w // 2)
            y4 = h - 1
            draw.polygon([(x1, y1), (x2, y2), (x3, y3), (x4, y4)],
                         fill=(0, 0, 0, transparency))
            img = Image.alpha_composite(img, overlay)
            img = img.convert('RGB')
            if isgray:
                img = ImageOps.grayscale(img)
            img = np.asarray(img).astype(np.uint8)
            return img

else:
    # Fallback placeholders if Albumentations is missing
    Erosion = None
    Dilation = None
    Bitmap = None
    Fog = None
    Frost = None
    Snow = None
    Rain = None
    Shadow = None


def clipped_zoom(img, zoom_factor):
    h = img.shape[1]
    ch = int(np.ceil(h / float(zoom_factor)))
    top = (h - ch) // 2
    img = scizoom(img[top:top + ch, top:top + ch],
                  (zoom_factor, zoom_factor, 1),
                  order=1)
    trim_top = (img.shape[0] - h) // 2
    return img[trim_top:trim_top + h, trim_top:trim_top + h]


def disk(radius, alias_blur=0.1, dtype=np.float32):
    if cv2 is None:
        return np.zeros((1, 1), dtype=dtype)
    if radius <= 8:
        coords = np.arange(-8, 8 + 1)
        ksize = (3, 3)
    else:
        coords = np.arange(-radius, radius + 1)
        ksize = (5, 5)
    x, y = np.meshgrid(coords, coords)
    aliased_disk = np.asarray((x**2 + y**2) <= radius**2, dtype=dtype)
    aliased_disk /= np.sum(aliased_disk)
    return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)


def plasma_fractal(mapsize=256, wibbledecay=3, rng=None):
    assert (mapsize & (mapsize - 1) == 0)
    maparray = np.empty((mapsize, mapsize), dtype=np.float64)
    maparray[0, 0] = 0
    stepsize = mapsize
    wibble = 100
    if rng is None:
        rng = np.random.default_rng()

    def wibbledmean(array):
        return array / 4 + wibble * rng.uniform(-wibble, wibble, array.shape)

    def fillsquares():
        cornerref = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        squareaccum = cornerref + np.roll(cornerref, shift=-1, axis=0)
        squareaccum += np.roll(squareaccum, shift=-1, axis=1)
        maparray[stepsize // 2:mapsize:stepsize,
                 stepsize // 2:mapsize:stepsize] = wibbledmean(squareaccum)

    def filldiamonds():
        drgrid = maparray[stepsize // 2:mapsize:stepsize,
                          stepsize // 2:mapsize:stepsize]
        ulgrid = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        ldrsum = drgrid + np.roll(drgrid, 1, axis=0)
        lulsum = ulgrid + np.roll(ulgrid, -1, axis=1)
        ltsum = ldrsum + lulsum
        maparray[0:mapsize:stepsize,
                 stepsize // 2:mapsize:stepsize] = wibbledmean(ltsum)
        tdrsum = drgrid + np.roll(drgrid, 1, axis=1)
        tulsum = ulgrid + np.roll(ulgrid, -1, axis=0)
        ttsum = tdrsum + tulsum
        maparray[stepsize // 2:mapsize:stepsize,
                 0:mapsize:stepsize] = wibbledmean(ttsum)

    while stepsize >= 2:
        fillsquares()
        filldiamonds()
        stepsize //= 2
        wibble /= wibbledecay
    maparray -= maparray.min()
    return maparray / maparray.max()


def motion_blur(img: np.ndarray, radius: int, sigma: float,
                angle: float) -> np.ndarray:
    if cv2 is None:
        return img
    kernel_size = max(1, int(radius) * 2 + 1)
    psf = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    psf[kernel_size // 2] = 1.0
    M = cv2.getRotationMatrix2D((kernel_size / 2, kernel_size / 2), angle, 1)
    psf = cv2.warpAffine(psf, M, (kernel_size, kernel_size))
    if sigma > 0:
        psf = cv2.GaussianBlur(psf, (kernel_size, kernel_size), sigma)
    psf /= psf.sum() if psf.sum() != 0 else 1
    return cv2.filter2D(img, -1, psf, borderType=cv2.BORDER_REPLICATE)


class CMERImageProcessor(BaseImageProcessor):
    model_input_names = [
        'pixel_values', 'orig_spatial_shape', 'expanded_from_indices',
        'is_original_flags'
    ]

    def __init__(
        self,
        down_sample_ratio: int = 32,
        do_convert_rgb: bool = True,
        do_rescale: bool = True,
        rescale_factor: float = 1.0 / 255.0,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, list[float]]] = None,
        image_std: Optional[Union[float, list[float]]] = None,
        resample: 'PILImageResampling' = PILImageResampling.BILINEAR,
        output_channel_format: ChannelDimension = ChannelDimension.FIRST,
        pad_value_strategy: str = 'mean',
        pad_value: Optional[Union[float, List[float]]] = None,
        center_pad: bool = False,
        do_augment: bool = True,
        augment_prob: float = 1.0,
        pre_pad_expand_ratio: float = 0.04,
        pre_pad_min_px: int = 8,
        aug_repeats: int = 0,
        keep_original: bool = True,
        num_workers: int = 8,
        pad_num_workers: Optional[int] = None,
        resize_backend: str = 'auto',
        normalize_inplace: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.down_sample_ratio = int(down_sample_ratio)
        self.do_convert_rgb = bool(do_convert_rgb)
        self.do_rescale = bool(do_rescale)
        self.rescale_factor = float(rescale_factor)
        self.do_normalize = bool(do_normalize)
        self.image_mean = image_mean if image_mean is not None else [
            0.5, 0.5, 0.5
        ]
        self.image_std = image_std if image_std is not None else [
            0.5, 0.5, 0.5
        ]
        self.resample = resample
        self.output_channel_format = output_channel_format
        self.pad_value_strategy = str(pad_value_strategy).lower()
        self.pad_value = pad_value
        self.center_pad = bool(center_pad)
        self.default_do_augment = bool(do_augment)
        self.augment_prob = float(augment_prob)
        self.pre_pad_expand_ratio = float(pre_pad_expand_ratio)
        self.pre_pad_min_px = int(pre_pad_min_px)
        self.aug_repeats = max(int(aug_repeats), 0)
        self.keep_original = bool(keep_original)
        self.num_workers = max(int(num_workers), 0)
        self.pad_num_workers = pad_num_workers if pad_num_workers is not None else self.num_workers
        self.resize_backend = resize_backend
        self.normalize_inplace = bool(normalize_inplace)
        self._augmentations = self._build_augmentations()

    def _build_augmentations(self):
        if A is None:
            logger.warning_once(
                f"[CMERImageProcessor] Albumentations 未安装，跳过图像增强。{_A_IMPORT_ERR if '_A_IMPORT_ERR' in globals() else ''}"
            )
            return None
        tlist = []
        if Bitmap is not None:
            tlist.append(Bitmap(p=0.2))
        weather_ops = []
        for op in (Fog, Frost, Snow, Rain, Shadow):
            if op is not None:
                try:
                    weather_ops.append(op())
                except Exception:
                    pass
        if weather_ops:
            tlist.append(A.OneOf(weather_ops, p=0.5))
        morph_ops = []
        if Erosion is not None:
            try:
                morph_ops.append(Erosion((2, 3)))
            except Exception:
                pass
        if Dilation is not None:
            try:
                morph_ops.append(Dilation((2, 3)))
            except Exception:
                pass
        if morph_ops:
            tlist.append(A.OneOf(morph_ops, p=0.2))
        tlist.extend([
            A.ShiftScaleRotate(shift_limit=0,
                               scale_limit=(-.15, 0),
                               rotate_limit=1,
                               border_mode=0,
                               interpolation=3,
                               value=[255, 255, 255],
                               p=1),
            A.GridDistortion(distort_limit=0.1,
                             border_mode=0,
                             interpolation=3,
                             value=[255, 255, 255],
                             p=0.5),
            A.RGBShift(r_shift_limit=15,
                       g_shift_limit=15,
                       b_shift_limit=15,
                       p=0.3),
            A.GaussNoise(var_limit=(10.0, 20.0), p=0.2),
            A.RandomBrightnessContrast(0.05, (-0.2, 0), True, p=0.2),
        ])
        return A.Compose(tlist, p=self.augment_prob)

    @staticmethod
    def _constant_border(img: np.ndarray,
                         pad_px: int,
                         value: int = 255) -> np.ndarray:
        if pad_px <= 0:
            return img
        if cv2 is not None:
            return cv2.copyMakeBorder(img,
                                      pad_px,
                                      pad_px,
                                      pad_px,
                                      pad_px,
                                      cv2.BORDER_CONSTANT,
                                      value=[value, value, value])
        return np.pad(img, ((pad_px, pad_px), (pad_px, pad_px), (0, 0)),
                      constant_values=value)

    def _maybe_augment_uint8(self, img_uint8: np.ndarray, seed: Optional[int],
                             pre_pad_px: int) -> np.ndarray:
        if self._augmentations is None:
            return img_uint8
        if pre_pad_px > 0:
            img_uint8 = self._constant_border(img_uint8, pre_pad_px, value=255)
        if seed is not None:
            rng_state = np.random.get_state()
            np.random.seed(seed)
            try:
                out = self._augmentations(image=img_uint8)['image']
            finally:
                np.random.set_state(rng_state)
            return out
        else:
            return self._augmentations(image=img_uint8)['image']

    def _prep_uint8(self, img, input_data_format) -> np.ndarray:
        if self.do_convert_rgb:
            img = convert_to_rgb(img)
        np_img = to_numpy_array(img)
        if input_data_format is None:
            _fmt = infer_channel_dimension_format(np_img)
        else:
            _fmt = input_data_format
        if _fmt == ChannelDimension.FIRST:
            np_img = np.transpose(np_img, (1, 2, 0))
        elif _fmt == ChannelDimension.LAST:
            pass
        else:
            np_img = to_channel_dimension_format(np_img,
                                                 ChannelDimension.LAST,
                                                 input_channel_dim=_fmt)
        if np_img.dtype != np.uint8:
            if np_img.dtype.kind == 'f':
                np_img = np.clip(np_img, 0.0, 1.0)
                np_img = (np_img * 255.0 + 0.5).astype(np.uint8)
            else:
                np_img = np_img.astype(np.uint8)
        return np_img

    def _resize_uint8(self, img_uint8: np.ndarray, th: int, tw: int,
                      backend: str) -> np.ndarray:
        if backend == 'cv2' and cv2 is not None:
            return cv2.resize(img_uint8, (tw, th), interpolation=1)
        return resize(img_uint8,
                      size=(th, tw),
                      resample=self.resample,
                      input_data_format=ChannelDimension.LAST)

    def preprocess_auto(self,
                        images: ImageInput,
                        return_tensors: Optional[Union[str,
                                                       TensorType]] = None,
                        trainer=None,
                        **kwargs) -> BatchFeature:
        if trainer is not None and getattr(trainer, 'is_in_train', False):
            kwargs.setdefault('do_augment', True)
        return self.preprocess(images, return_tensors=return_tensors, **kwargs)

    @filter_out_non_signature_kwargs()
    def preprocess(
        self,
        images: ImageInput,
        return_tensors: Optional[Union[str, TensorType]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        do_convert_rgb: Optional[bool] = None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, list[float]]] = None,
        image_std: Optional[Union[float, list[float]]] = None,
        down_sample_ratio: Optional[int] = None,
        resample: Optional['PILImageResampling'] = None,
        output_channel_format: Optional[ChannelDimension] = None,
        pad_value_strategy: Optional[str] = None,
        pad_value: Optional[Union[float, List[float]]] = None,
        center_pad: Optional[bool] = None,
        do_augment: Optional[bool] = True,
        augment_seed: Optional[int] = None,
        pre_pad_expand_ratio: Optional[float] = None,
        pre_pad_min_px: Optional[int] = None,
        aug_repeats: Optional[int] = None,
        keep_original: Optional[bool] = None,
        num_workers: Optional[int] = None,
        pad_num_workers: Optional[int] = None,
        resize_backend: Optional[str] = None,
        normalize_inplace: Optional[bool] = None,
    ) -> BatchFeature:
        do_convert_rgb = self.do_convert_rgb if do_convert_rgb is None else do_convert_rgb
        do_rescale = self.do_rescale if do_rescale is None else do_rescale
        rescale_factor = self.rescale_factor if rescale_factor is None else rescale_factor
        do_normalize = self.do_normalize if do_normalize is None else do_normalize
        image_mean = self.image_mean if image_mean is None else image_mean
        image_std = self.image_std if image_std is None else image_std
        down_sample_ratio = self.down_sample_ratio if down_sample_ratio is None else int(
            down_sample_ratio)
        resample = self.resample if resample is None else resample
        output_channel_format = self.output_channel_format if output_channel_format is None else output_channel_format
        pad_value_strategy = self.pad_value_strategy if pad_value_strategy is None else pad_value_strategy.lower(
        )
        pad_value = self.pad_value if pad_value is None else pad_value
        center_pad = self.center_pad if center_pad is None else bool(
            center_pad)
        do_augment = self.default_do_augment if do_augment is None else bool(
            do_augment)
        pre_pad_expand_ratio = self.pre_pad_expand_ratio if pre_pad_expand_ratio is None else float(
            pre_pad_expand_ratio)
        pre_pad_min_px = self.pre_pad_min_px if pre_pad_min_px is None else int(
            pre_pad_min_px)
        aug_repeats = self.aug_repeats if aug_repeats is None else max(
            int(aug_repeats), 0)
        keep_original = self.keep_original if keep_original is None else bool(
            keep_original)
        num_workers = self.num_workers if num_workers is None else max(
            int(num_workers), 0)
        pad_num_workers = self.pad_num_workers if pad_num_workers is None else max(
            int(pad_num_workers), 0)
        resize_backend = (self.resize_backend if resize_backend is None else
                          resize_backend).lower()
        normalize_inplace = self.normalize_inplace if normalize_inplace is None else bool(
            normalize_inplace)
        if type(images) is dict:
            images = images.get('image', None)
            images = self.fetch_images(images)
        else:
            images = self.fetch_images(images)
        images = make_flat_list_of_images(images)
        if not valid_images(images):
            raise ValueError(
                'Invalid image type. Must be PIL.Image.Image, numpy.ndarray, or torch.Tensor'
            )
        validate_preprocess_arguments(
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
        )
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _process_one(idx_img: int):
            base = self._prep_uint8(images[idx_img], input_data_format)
            h0, w0 = base.shape[:2]
            results_imgs: List[np.ndarray] = []
            results_sizes: List[Tuple[int, int]] = []
            results_from: List[int] = []
            results_flag: List[bool] = []
            cand: List[Tuple[np.ndarray, bool]] = []
            if keep_original:
                cand.append((base, True))
            if do_augment and self._augmentations is not None and aug_repeats > 0:
                est_pad = max(
                    int(max(h0, w0) * pre_pad_expand_ratio),
                    pre_pad_min_px if pre_pad_expand_ratio > 0 else 0)
                for k in range(aug_repeats):
                    seed_k = None if augment_seed is None else (
                        int(augment_seed) + idx_img * (aug_repeats + 1) + k)
                    aug_img = self._maybe_augment_uint8(base,
                                                        seed=seed_k,
                                                        pre_pad_px=est_pad)
                    cand.append((aug_img, False))

            is_cv2_avail = (resize_backend == 'auto' and cv2 is not None)
            be = 'cv2' if is_cv2_avail else resize_backend
            max_long_edge = 1024

            for uint8_img, is_orig in cand:
                hh, ww = uint8_img.shape[:2]
                if max(hh, ww) > max_long_edge:
                    scale = float(max_long_edge) / float(max(hh, ww))
                    targ_h = max(1, int(math.floor(hh * scale)))
                    targ_w = max(1, int(math.floor(ww * scale)))
                    uint8_img = self._resize_uint8(uint8_img, targ_h, targ_w,
                                                   be)
                    hh, ww = uint8_img.shape[:2]
                MIN_HW = 224
                ds = down_sample_ratio
                ceil_h = max(MIN_HW, math.ceil(hh / ds) * ds)
                ceil_w = max(MIN_HW, math.ceil(ww / ds) * ds)
                if max(ceil_h, ceil_w) <= max_long_edge:
                    th, tw = ceil_h, ceil_w
                else:
                    floor_h = max(MIN_HW, (hh // ds) * ds)
                    floor_w = max(MIN_HW, (ww // ds) * ds)
                    if floor_h <= 0 or floor_w <= 0:
                        floor_h = max(MIN_HW,
                                      min(hh, max_long_edge) // ds * ds)
                        floor_w = max(MIN_HW,
                                      min(ww, max_long_edge) // ds * ds)
                    th, tw = floor_h, floor_w

                rs_img = self._resize_uint8(uint8_img, th, tw, be)
                if do_rescale:
                    rs_img = rs_img.astype(np.float32)
                    np.multiply(rs_img,
                                float(rescale_factor),
                                out=rs_img,
                                casting='unsafe')
                else:
                    rs_img = rs_img.astype(np.float32)
                results_imgs.append(rs_img)
                results_sizes.append((th, tw))
                results_from.append(idx_img)
                results_flag.append(is_orig)
            return results_imgs, results_sizes, results_from, results_flag

        proc_list: List[np.ndarray] = []
        rec_sizes: List[Tuple[int, int]] = []
        from_indices: List[int] = []
        is_orig_flags: List[bool] = []
        if num_workers and num_workers > 1 and len(images) > 1:
            with ThreadPoolExecutor(max_workers=num_workers) as ex:
                futs = [ex.submit(_process_one, i) for i in range(len(images))]
                for fu in as_completed(futs):
                    imgs_i, sizes_i, from_i, flag_i = fu.result()
                    proc_list.extend(imgs_i)
                    rec_sizes.extend(sizes_i)
                    from_indices.extend(from_i)
                    is_orig_flags.extend(flag_i)
        else:
            for i in range(len(images)):
                imgs_i, sizes_i, from_i, flag_i = _process_one(i)
                proc_list.extend(imgs_i)
                rec_sizes.extend(sizes_i)
                from_indices.extend(from_i)
                is_orig_flags.extend(flag_i)
        if len(proc_list) == 0:
            return BatchFeature(data={
                'image': [],
                'orig_spatial_shape': [],
                'expanded_from_indices': [],
                'is_original_flags': []
            },
                                tensor_type=return_tensors)
        max_h = max(h for h, _ in rec_sizes)
        max_w = max(w for _, w in rec_sizes)
        mean = np.array(image_mean, dtype=np.float32)
        std = np.array(image_std, dtype=np.float32)
        inv_std = 1.0 / np.where(std == 0, 1.0, std)

        def _maybe_scale_stats_to_image_domain(
                _arr: np.ndarray, exemplar: np.ndarray) -> np.ndarray:
            if not do_rescale and exemplar.max() > 1.5 and _arr.max() <= 1.5:
                return _arr * 255.0
            return _arr

        def _make_pad_color(c: int, exemplar: np.ndarray) -> np.ndarray:
            _mean = _maybe_scale_stats_to_image_domain(mean, exemplar)
            if pad_value_strategy == 'mean':
                col = _mean
            elif pad_value_strategy == 'white':
                col = np.ones(
                    (c, ), dtype=np.float32) * (1.0 if do_rescale else 255.0)
            elif pad_value_strategy == 'zero':
                col = np.zeros((c, ), dtype=np.float32)
            elif pad_value_strategy == 'custom':
                if pad_value is None:
                    col = _mean
                else:
                    col = np.array(pad_value, dtype=np.float32)
                    if col.ndim == 0:
                        col = np.full((c, ), float(col), dtype=np.float32)
                    if col.shape[0] != c:
                        raise ValueError(
                            f'pad_value length must match channels={c}')
            else:
                col = _mean
            return col

        def _to_ch_first(arr: np.ndarray) -> np.ndarray:
            return np.transpose(arr, (2, 0, 1))

        batched: List[np.ndarray] = [None] * len(proc_list)

        def _pad_one(i: int):
            np_img = proc_list[i]
            h, w = rec_sizes[i]
            C = np_img.shape[2]
            pad_color = _make_pad_color(C, np_img)
            if center_pad:
                y0 = (max_h - h) // 2
                x0 = (max_w - w) // 2
            else:
                y0 = 0
                x0 = 0
            pad_img = np.empty((max_h, max_w, C), dtype=np.float32)
            pad_img[...] = pad_color
            pad_img[y0:y0 + h, x0:x0 + w, :] = np_img
            if do_normalize:
                _mean = _maybe_scale_stats_to_image_domain(mean, pad_img)
                _invstd = _maybe_scale_stats_to_image_domain(inv_std, pad_img)
                if normalize_inplace:
                    np.subtract(pad_img, _mean, out=pad_img)
                    np.multiply(pad_img, _invstd, out=pad_img)
                else:
                    pad_img = (pad_img - _mean) * _invstd
            batched[i] = _to_ch_first(
                pad_img
            ) if output_channel_format == ChannelDimension.FIRST else pad_img

        if pad_num_workers and pad_num_workers > 1 and len(proc_list) > 1:
            with ThreadPoolExecutor(max_workers=pad_num_workers) as ex:
                list(ex.map(_pad_one, range(len(proc_list))))
        else:
            for i in range(len(proc_list)):
                _pad_one(i)
        return BatchFeature(
            data={
                'image': batched,
                'orig_spatial_shape': rec_sizes,
                'expanded_from_indices': from_indices,
                'is_original_flags': is_orig_flags,
            },
            tensor_type=return_tensors,
        )


AutoImageProcessor.register('CMER',
                            slow_image_processor_class=CMERImageProcessor)
class CMERProcessor(ProcessorMixin):
    attributes = ['image_processor', 'tokenizer']
    image_processor_class = 'CMERImageProcessor'
    tokenizer_class = 'PreTrainedTokenizerFast'

    def __init__(
        self, 
        image_processor=None, 
        tokenizer=None, 
        tokenizer_file: str = './configs/rec/cmer/cmer_tokenizer/tokenizer.json',
        **kwargs
    ):
        if image_processor is None:
            # 确保这里能正确导入你的 CMERImageProcessor
            image_processor = CMERImageProcessor(**kwargs)

        if tokenizer is None:
            try:
                tokenizer = PreTrainedTokenizerFast(
                    tokenizer_file=tokenizer_file,
                    padding_side="right",
                    truncation_side="right",
                    pad_token="<|pad|>",
                    bos_token="<|bos|>",
                    eos_token="<|eos|>",
                    unk_token="<|unk|>",
                )
            except Exception as e:
                # logger 需要外部定义或引入，这里简单用 print 代替
                print(f"Failed to initialize default tokenizer from {tokenizer_file}. Error: {e}")
                tokenizer = None

        super().__init__(image_processor=image_processor, tokenizer=tokenizer)


    def __call__(
        self,
        images: ImageInput,
        text: Union[str, List[str]]=None,
        ids=None,
        categorys=None,
        return_tensors: Optional[Union[str, TensorType]] = "pt",
        padding: Union[bool, str] = True,
        truncation: bool = True,
        max_length: Optional[int] = None,
        **img_kwargs,
    ):  
        if isinstance(images, dict) and "image" in images:
            images = images["image"]
        # 情况 2: 列表样本，例如 [{'image': <PIL...>}, {'image': <PIL...>}]
        elif isinstance(images, (list, tuple)) and len(images) > 0 and isinstance(images[0], dict) and "image" in images[0]:
            images = [img["image"] for img in images]
         # 计算输入图片的数量，用于后续生成默认 text
        if isinstance(images, (list, tuple)):
            input_batch_size = len(images)
        else:
            input_batch_size = 1
        image_outputs: BatchFeature = self.image_processor.preprocess(
            images=images,
            return_tensors=return_tensors,
            **img_kwargs,
        )
        expanded_from = image_outputs.get("expanded_from_indices")
                # =================================================================
        # 2. [修复核心报错] 处理 text/ids/categorys 为 None 的情况
        # =================================================================
        # 如果 text 为 None (推理模式)，生成空字符串列表
        if text is None:
            text_list = [""] * input_batch_size
        elif isinstance(text, str):
            text_list = [text]
        else:
            text_list = list(text)

        # 如果 ids 为 None，生成默认占位符
        if ids is None:
            ids_list = [None] * len(text_list)
        else:
            ids_list = list(ids)

        # 如果 categorys 为 None，生成默认占位符
        if categorys is None:
            cats_list = [None] * len(text_list)
        else:
            cats_list = list(categorys)
        # =================================================================

        if expanded_from is None:
            num_in = len(text_list)
            expanded_from = list(range(num_in))
        else:
            num_in = max(expanded_from) + 1
            
        # 检查长度一致性
        if not (len(text_list) == num_in == len(ids_list) == len(cats_list)):
            raise ValueError(
                f"[CMERProcessor] Mismatch between base counts: "
                f"text={len(text_list)}, ids={len(ids_list)}, "
                f"cats={len(cats_list)}, num_in(from expanded_from)={num_in}"
            )
            
        bos_token = self.tokenizer.bos_token
        eos_token = self.tokenizer.eos_token
        if bos_token is None or eos_token is None:
            raise ValueError("Tokenizer must have a `bos_token` and an `eos_token`.")
            
        base_texts = text_list
        base_ids = ids_list
        base_cats = cats_list
        
        try:
            expanded_texts = [
                f"{bos_token}{base_texts[src]}{eos_token}" for src in expanded_from
            ]
            expanded_ids = [base_ids[src] for src in expanded_from]
            expanded_cats = [base_cats[src] for src in expanded_from]
        except IndexError:
            raise ValueError(
                f"[CMERProcessor] expanded_from_indices contains index out of range: "
                f"max={max(expanded_from)}, but num_in={num_in}"
            )
            
        text_outputs = self.tokenizer(
            expanded_texts,
            return_tensors=return_tensors,
            add_special_tokens=True,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
        )
        
        text_outputs["decoder_input_ids"] = text_outputs.pop("input_ids")
        data = {**image_outputs, **text_outputs}
        
        labels = (
            data["decoder_input_ids"].clone()
            if return_tensors is not None
            else list(data["decoder_input_ids"])
        )
        
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            data["labels"] = labels
        else:
            if hasattr(labels, "masked_fill"):
                labels = labels.masked_fill(labels == pad_id, -100)
            else:
                labels = [[(-100 if tok == pad_id else tok) for tok in seq] for seq in labels]
            data["labels"] = labels

        # bf = BatchFeature(data=data, tensor_type=return_tensors)
        # bf["ids"] = expanded_ids
        # bf["categorys"] = expanded_cats
        input_ids = data["decoder_input_ids"]
        # return bf

        if "attention_mask" in text_outputs:
            # attention_mask shape: [batch, seq_len]
            # sum(dim=1) 得到每个样本的有效长度
            length = text_outputs["attention_mask"].sum(dim=1)
            # 确保是 int32 或 int64
            length = length.to(dtype=torch.int32)
        else:
            # 如果没有 attention_mask，假设没有 padding，直接取 shape
            seq_len = input_ids.shape[1]
            batch_size = input_ids.shape[0]
            length = torch.full((batch_size,), seq_len, dtype=torch.int32)

        # 6. 返回 Tuple (pixel_values, labels, length)
        pixel_values = data['image']    
        return pixel_values, labels, length

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)