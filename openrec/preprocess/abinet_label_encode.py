import numpy as np

from .ctc_label_encode import BaseRecLabelEncode


class ABINetLabelEncode(BaseRecLabelEncode):
    """Convert between text-label and text-index."""

    def __init__(self,
                 max_text_length,
                 character_dict_path=None,
                 use_space_char=False,
                 ignore_index=100,
                 **kwargs):

        super(ABINetLabelEncode,
              self).__init__(max_text_length, character_dict_path,
                             use_space_char)
        self.ignore_index = ignore_index

    def __call__(self, data):
        text = data['label']
        text = self.encode(text)
        if text is None:
            return None
        if len(text) > self.max_text_len:
            return None
        data['length'] = np.array(len(text))
        text.append(0)
        text = text + [self.ignore_index] * (self.max_text_len + 1 - len(text))
        data['label'] = np.array(text)
        return data

    def add_special_char(self, dict_character):
        dict_character = ['</s>'] + dict_character
        return dict_character
