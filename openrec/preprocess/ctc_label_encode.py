import re

import numpy as np

from tools.utils.logging import get_logger


class BaseRecLabelEncode(object):
    """Convert between text-label and text-index."""

    def __init__(
        self,
        max_text_length,
        character_dict_path=None,
        use_space_char=False,
        lower=False,
    ):
        self.max_text_len = max_text_length
        self.beg_str = 'sos'
        self.end_str = 'eos'
        self.lower = lower
        self.reverse = False
        if character_dict_path is None:
            logger = get_logger()
            logger.warning(
                'The character_dict_path is None, model can only recognize number and lower letters'
            )
            self.character_str = '0123456789abcdefghijklmnopqrstuvwxyz'
            dict_character = list(self.character_str)
            self.lower = True
        else:
            self.character_str = []
            with open(character_dict_path, 'rb') as fin:
                lines = fin.readlines()
                for line in lines:
                    line = line.decode('utf-8').strip('\n').strip('\r\n')
                    self.character_str.append(line)
            if use_space_char:
                self.character_str.append(' ')
            dict_character = list(self.character_str)
            if 'arabic' in character_dict_path:
                self.reverse = True
        dict_character = self.add_special_char(dict_character)
        self.dict = {}
        for i, char in enumerate(dict_character):
            self.dict[char] = i
        self.character = dict_character

    def label_reverse(self, text):
        text_re = []
        c_current = ''
        for c in text:
            if not bool(re.search('[a-zA-Z0-9 :*./%+-١٢٣٤٥٦٧٨٩٠]', c)):
                if c_current != '':
                    text_re.append(c_current)
                text_re.append(c)
                c_current = ''
            else:
                c_current += c
        if c_current != '':
            text_re.append(c_current)

        return ''.join(text_re[::-1])

    def add_special_char(self, dict_character):
        return dict_character

    def encode(self, text):
        """convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]

        output:
            text: concatenated text index for CTCLoss.
                    [sum(text_lengths)] = [text_index_0 + text_index_1 + ... + text_index_(n - 1)]
            length: length of each text. [batch_size]
        """
        if len(text) == 0:
            return None
        if self.lower:
            text = text.lower()
        text_list = []
        for char in text:
            if char not in self.dict:
                continue
            text_list.append(self.dict[char])
        if len(text_list) == 0 or len(text_list) > self.max_text_len:
            return None
        return text_list


class CTCLabelEncode(BaseRecLabelEncode):
    """Convert between text-label and text-index."""

    def __init__(self,
                 max_text_length,
                 character_dict_path=None,
                 use_space_char=False,
                 **kwargs):
        super(CTCLabelEncode,
              self).__init__(max_text_length, character_dict_path,
                             use_space_char)
        self.is_reverse = kwargs.get('is_reverse', False)

    def __call__(self, data):
        text = data['label']
        if self.reverse and self.is_reverse:  # for arabic rec
            text = self.label_reverse(text)
        text = self.encode(text)
        if text is None:
            return None
        data['length'] = np.array(len(text))
        text = text + [0] * (self.max_text_len - len(text))
        data['label'] = np.array(text)

        label = [0] * len(self.character)
        for x in text:
            label[x] += 1
        data['label_ace'] = np.array(label)
        return data

    def add_special_char(self, dict_character):
        dict_character = ['blank'] + dict_character
        return dict_character
