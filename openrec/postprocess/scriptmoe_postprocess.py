"""
PostProcess for Script-Aware MoE Decoder.

During inference, ScriptMoEDecoder returns standard logits (same as NRTRDecoder),
so we can reuse ARLabelDecode directly.

During training, ScriptMoEDecoder returns a dict with 'logit' key.
This wrapper extracts the logit from the dict and delegates to ARLabelDecode.
"""

import torch
import numpy as np

from .ar_postprocess import ARLabelDecode


class ScriptMoELabelDecode(ARLabelDecode):
    """Decode predictions from ScriptMoEDecoder.

    Handles both training (dict output) and inference (tensor output).
    """

    def __init__(self, character_dict_path=None, use_space_char=True, **kwargs):
        super().__init__(character_dict_path, use_space_char, **kwargs)

    def __call__(self, preds, batch=None, *args, **kwargs):
        # During training, preds is a dict; extract 'logit'
        if isinstance(preds, dict):
            preds = preds['logit']

        # During inference, preds is already a tensor
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()
        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)
        text = self.decode(preds_idx, preds_prob)
        if batch is None:
            return text
        label = batch[1]
        label = self.decode(label[:, 1:])
        return text, label
