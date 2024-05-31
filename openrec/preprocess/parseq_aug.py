# Scene Text Recognition Model Hub
# Copyright 2022 Darwin Bautista
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import partial

import imgaug.augmenters as iaa
import numpy as np
from PIL import Image, ImageFilter

from openrec.preprocess import auto_augment
from openrec.preprocess.auto_augment import _LEVEL_DENOM, LEVEL_TO_ARG, NAME_TO_OP, _randomly_negate, rotate


def rotate_expand(img, degrees, **kwargs):
    """Rotate operation with expand=True to avoid cutting off the
    characters."""
    kwargs['expand'] = True
    return rotate(img, degrees, **kwargs)


def _level_to_arg(level, hparams, key, default):
    magnitude = hparams.get(key, default)
    level = (level / _LEVEL_DENOM) * magnitude
    level = _randomly_negate(level)
    return level,


def apply():
    # Overrides
    NAME_TO_OP.update({'Rotate': rotate_expand})
    LEVEL_TO_ARG.update({
        'Rotate':
        partial(_level_to_arg, key='rotate_deg', default=30.),
        'ShearX':
        partial(_level_to_arg, key='shear_x_pct', default=0.3),
        'ShearY':
        partial(_level_to_arg, key='shear_y_pct', default=0.3),
        'TranslateXRel':
        partial(_level_to_arg, key='translate_x_pct', default=0.45),
        'TranslateYRel':
        partial(_level_to_arg, key='translate_y_pct', default=0.45),
    })


apply()

_OP_CACHE = {}


def _get_op(key, factory):
    try:
        op = _OP_CACHE[key]
    except KeyError:
        op = factory()
        _OP_CACHE[key] = op
    return op


def _get_param(level, img, max_dim_factor, min_level=1):
    max_level = max(min_level, max_dim_factor * max(img.size))
    return round(min(level, max_level))


def gaussian_blur(img, radius, **__):
    radius = _get_param(radius, img, 0.02)
    key = 'gaussian_blur_' + str(radius)
    op = _get_op(key, lambda: ImageFilter.GaussianBlur(radius))
    return img.filter(op)


def motion_blur(img, k, **__):
    k = _get_param(k, img, 0.08, 3) | 1  # bin to odd values
    key = 'motion_blur_' + str(k)
    op = _get_op(key, lambda: iaa.MotionBlur(k))
    return Image.fromarray(op(image=np.asarray(img)))


def gaussian_noise(img, scale, **_):
    scale = _get_param(scale, img, 0.25) | 1  # bin to odd values
    key = 'gaussian_noise_' + str(scale)
    op = _get_op(key, lambda: iaa.AdditiveGaussianNoise(scale=scale))
    return Image.fromarray(op(image=np.asarray(img)))


def poisson_noise(img, lam, **_):
    lam = _get_param(lam, img, 0.2) | 1  # bin to odd values
    key = 'poisson_noise_' + str(lam)
    op = _get_op(key, lambda: iaa.AdditivePoissonNoise(lam))
    return Image.fromarray(op(image=np.asarray(img)))


def _level_to_arg(level, _hparams, max):
    level = max * level / auto_augment._LEVEL_DENOM
    return level,


_RAND_TRANSFORMS = auto_augment._RAND_INCREASING_TRANSFORMS.copy()
_RAND_TRANSFORMS.remove(
    'SharpnessIncreasing')  # remove, interferes with *blur ops
_RAND_TRANSFORMS.extend([
    'GaussianBlur',
    # 'MotionBlur',
    # 'GaussianNoise',
    'PoissonNoise'
])
auto_augment.LEVEL_TO_ARG.update({
    'GaussianBlur':
    partial(_level_to_arg, max=4),
    'MotionBlur':
    partial(_level_to_arg, max=20),
    'GaussianNoise':
    partial(_level_to_arg, max=0.1 * 255),
    'PoissonNoise':
    partial(_level_to_arg, max=40)
})
auto_augment.NAME_TO_OP.update({
    'GaussianBlur': gaussian_blur,
    'MotionBlur': motion_blur,
    'GaussianNoise': gaussian_noise,
    'PoissonNoise': poisson_noise
})


def rand_augment_transform(magnitude=5, num_layers=3):
    # These are tuned for magnitude=5, which means that effective magnitudes are half of these values.
    hparams = {
        'rotate_deg': 30,
        'shear_x_pct': 0.9,
        'shear_y_pct': 0.2,
        'translate_x_pct': 0.10,
        'translate_y_pct': 0.30
    }
    ra_ops = auto_augment.rand_augment_ops(magnitude,
                                           hparams=hparams,
                                           transforms=_RAND_TRANSFORMS)
    # Supply weights to disable replacement in random selection (i.e. avoid applying the same op twice)
    choice_weights = [1. / len(ra_ops) for _ in range(len(ra_ops))]
    return auto_augment.RandAugment(ra_ops, num_layers, choice_weights)
