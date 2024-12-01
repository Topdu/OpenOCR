import copy

from .base_detector import BaseDetector

__all__ = ['build_model']


def build_model(config):
    config = copy.deepcopy(config)
    det_model = BaseDetector(config)
    return det_model
