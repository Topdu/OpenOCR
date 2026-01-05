import copy

from .base_recognizer import BaseRecognizer
from .cmer_modeling.modeling_cmer import build_model_cmer
__all__ = ['build_model','build_cmer_model']


def build_model(config):
    config = copy.deepcopy(config)
    rec_model = BaseRecognizer(config)
    return rec_model

def build_cmer_model(config):
    config = copy.deepcopy(config)
    rec_model = build_model_cmer(config)
    return rec_model