from random import sample

import numpy as np

from .ctc_label_encode import BaseRecLabelEncode


class VisionLANLabelEncode(BaseRecLabelEncode):
    """Convert between text-label and text-index."""

    def __init__(self,
                 max_text_length,
                 character_dict_path=None,
                 use_space_char=False,
                 **kwargs):
        super(VisionLANLabelEncode,
              self).__init__(max_text_length, character_dict_path,
                             use_space_char)
        self.dict = {}
        for i, char in enumerate(self.character):
            self.dict[char] = i

    def __call__(self, data):
        text = data['label']  # original string
        # generate occluded text
        len_str = len(text)
        if len_str <= 0:
            return None
        change_num = 1
        order = list(range(len_str))
        change_id = sample(order, change_num)[0]
        label_sub = text[change_id]
        if change_id == (len_str - 1):
            label_res = text[:change_id]
        elif change_id == 0:
            label_res = text[1:]
        else:
            label_res = text[:change_id] + text[change_id + 1:]

        data['label_res'] = label_res  # remaining string
        data['label_sub'] = label_sub  # occluded character
        data['label_id'] = change_id  # character index
        # encode label
        text = self.encode(text)
        if text is None:
            return None
        text = [i + 1 for i in text]
        data['length'] = np.array(len(text))
        text = text + [0] * (self.max_text_len + 1 - len(text))
        data['label'] = np.array(text)
        label_res = self.encode(label_res)
        label_sub = self.encode(label_sub)
        if label_res is None:
            label_res = []
        else:
            label_res = [i + 1 for i in label_res]
        if label_sub is None:
            label_sub = []
        else:
            label_sub = [i + 1 for i in label_sub]
        data['length_res'] = np.array(len(label_res))
        data['length_sub'] = np.array(len(label_sub))
        label_res = label_res + [0] * (self.max_text_len - len(label_res))
        label_sub = label_sub + [0] * (self.max_text_len - len(label_sub))
        data['label_res'] = np.array(label_res)
        data['label_sub'] = np.array(label_sub)
        return data
