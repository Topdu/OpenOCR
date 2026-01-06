import numpy as np
from openrec.preprocess.ctc_label_encode import BaseRecLabelEncode

import os

os.environ['TOKENIZERS_PARALLELISM'] = 'false'


class UniRecLabelEncode(BaseRecLabelEncode):
    """ Convert between text-label and text-index """
    SPACE = '[s]'
    GO = '[GO]'
    list_token = [GO, SPACE]

    def __init__(self,
                 max_text_length,
                 character_dict_path=None,
                 use_space_char=False,
                 vlmocr=False,
                 tokenizer_path='./galactica',
                 **kwargs):
        super(UniRecLabelEncode,
              self).__init__(max_text_length, character_dict_path,
                             use_space_char)
        # character (str): set of the possible characters.
        # [GO] for the start token of the attention decoder. [s] for end-of-sentence token.
        if vlmocr:
            self.padding_idx = 1
            self.eos_idx = 2
            self.bos_idx = 0
        else:
            self.padding_idx = 0
            self.eos_idx = 2
            self.bos_idx = 1
        self.batch_max_length = max_text_length + 3
        from transformers import AutoTokenizer  # transformers==4.2.1
        self.bpe_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    def __call__(self, data):
        text = data['label']
        bpe_text, length = self.bpe_encode(text)
        if bpe_text is None:
            return None
        data['label'] = np.array(bpe_text)
        data['length'] = np.array(length)
        return data

    def add_special_char(self, dict_character):
        dict_character = self.list_token + dict_character
        return dict_character

    def bpe_encode(self, text):
        if len(text) == 0:
            return None, None
        token = self.bpe_tokenizer(text)['input_ids']
        length = len(token)
        text_list = [self.bos_idx] + token + [2]
        if len(text_list) == 0 or len(text_list) > self.batch_max_length:
            return None, None
        text_list = text_list + [self.padding_idx
                                 ] * (self.batch_max_length - len(text_list))
        return text_list, length
