import numpy as np

from .ctc_label_encode import BaseRecLabelEncode


class CharLabelEncode(BaseRecLabelEncode):
    """Convert between text-label and text-index."""

    def __init__(self,
                 max_text_length,
                 character_dict_path=None,
                 use_space_char=False,
                 **kwargs):
        super(CharLabelEncode,
              self).__init__(max_text_length, character_dict_path,
                             use_space_char)

    def __call__(self, data):
        text = data['label']
        text = self.encode(text)
        if text is None:
            return None
        if len(text) > self.max_text_len:
            return None
        data['length'] = np.array(len(text))
        text_char = text + [104] * (self.max_text_len + 1 - len(text))
        text.insert(0, 2)
        text.append(3)
        text = text + [0] * (self.max_text_len + 2 - len(text))
        data['label'] = np.array(text)
        data['label_char'] = np.array(text_char)
        return data

    def add_special_char(self, dict_character):
        dict_character = ['blank', '<unk>', '<s>', '</s>'] + dict_character
        return dict_character
