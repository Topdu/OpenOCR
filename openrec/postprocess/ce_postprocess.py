import torch

from .ctc_postprocess import BaseRecLabelDecode


class CELabelDecode(BaseRecLabelDecode):
    """Convert between text-label and text-index."""

    def __init__(self,
                 character_dict_path=None,
                 use_space_char=False,
                 **kwargs):
        super(CELabelDecode, self).__init__(character_dict_path,
                                            use_space_char)

    def __call__(self, preds, label=None, *args, **kwargs):
        if isinstance(preds, tuple) or isinstance(preds, list):
            preds = preds[-1]
        if isinstance(preds, torch.Tensor):
            preds = preds.numpy()
        preds_idx = preds.argmax(axis=-1)
        preds_prob = preds.max(axis=-1)
        text = self.decode(preds_idx, preds_prob)
        if label is None:
            return text
        label = self.decode(label.flatten())
        return text, label

    def decode(self, text_index, text_prob=None):
        """convert text-index into text-label."""
        result_list = []
        batch_size = len(text_index)
        for batch_idx in range(batch_size):
            text = self.character[text_index[batch_idx]]
            if text_prob is not None:
                conf_list = text_prob[batch_idx]
            else:
                conf_list = 1.0
            result_list.append((text, conf_list))
        return result_list

    def add_special_char(self, dict_character):
        return dict_character
