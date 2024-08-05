import numpy as np

from .ce_label_encode import BaseRecLabelEncode


class SRNLabelEncode(BaseRecLabelEncode):
    """Convert between text-label and text-index."""

    def __init__(self,
                 max_text_length,
                 character_dict_path=None,
                 use_space_char=False,
                 **kwargs):
        super(SRNLabelEncode,
              self).__init__(max_text_length, character_dict_path,
                             use_space_char)

    def add_special_char(self, dict_character):
        dict_character = dict_character + ['<BOS>', '<EOS>']
        self.start_idx = len(dict_character) - 2
        self.end_idx = len(dict_character) - 1
        return dict_character

    def __call__(self, data):
        text = data['label']
        text = self.encode(text)
        if text is None:
            return None
        if len(text) > self.max_text_len:
            return None
        data['length'] = np.array(len(text))
        text = text + [self.end_idx] * (self.max_text_len - len(text))
        data['label'] = np.array(text)
        return data

    def get_ignored_tokens(self):
        return [self.start_idx, self.end_idx]
