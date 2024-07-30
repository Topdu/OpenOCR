'''
This code is refer from:
https://github.com/AlibabaResearch/AdvancedLiterateMachinery/blob/main/OCR/MGP-STR
'''
import numpy as np

from openrec.preprocess.ctc_label_encode import BaseRecLabelEncode


class MGPLabelEncode(BaseRecLabelEncode):
    """ Convert between text-label and text-index """
    SPACE = '[s]'
    GO = '[GO]'
    list_token = [GO, SPACE]

    def __init__(self,
                 max_text_length,
                 character_dict_path=None,
                 use_space_char=False,
                 only_char=False,
                 **kwargs):
        super(MGPLabelEncode,
              self).__init__(max_text_length, character_dict_path,
                             use_space_char)
        # character (str): set of the possible characters.
        # [GO] for the start token of the attention decoder. [s] for end-of-sentence token.

        self.batch_max_length = max_text_length + len(self.list_token)
        self.only_char = only_char
        if not only_char:
            # transformers==4.2.1
            from transformers import BertTokenizer, GPT2Tokenizer
            self.bpe_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            self.wp_tokenizer = BertTokenizer.from_pretrained(
                'bert-base-uncased')

    def __call__(self, data):
        text = data['label']
        char_text, char_len = self.encode(text)
        if char_text is None:
            return None
        data['length'] = np.array(char_len)
        data['char_label'] = np.array(char_text)
        if self.only_char:
            return data
        bpe_text = self.bpe_encode(text)
        if bpe_text is None:
            return None
        wp_text = self.wp_encode(text)
        data['bpe_label'] = np.array(bpe_text)
        data['wp_label'] = wp_text
        return data

    def add_special_char(self, dict_character):
        dict_character = self.list_token + dict_character
        return dict_character

    def encode(self, text):
        """ convert text-label into text-index.
        """
        if len(text) == 0:
            return None, None
        if self.lower:
            text = text.lower()
        length = len(text)
        text = [self.GO] + list(text) + [self.SPACE]
        text_list = []
        for char in text:
            if char not in self.dict:
                continue
            text_list.append(self.dict[char])
        if len(text_list) == 0 or len(text_list) > self.batch_max_length:
            return None, None
        text_list = text_list + [self.dict[self.GO]
                                 ] * (self.batch_max_length - len(text_list))
        return text_list, length

    def bpe_encode(self, text):
        if len(text) == 0:
            return None
        token = self.bpe_tokenizer(text)['input_ids']
        text_list = [1] + token + [2]
        if len(text_list) == 0 or len(text_list) > self.batch_max_length:
            return None
        text_list = text_list + [self.dict[self.GO]
                                 ] * (self.batch_max_length - len(text_list))
        return text_list

    def wp_encode(self, text):
        wp_target = self.wp_tokenizer([text],
                                      padding='max_length',
                                      max_length=self.batch_max_length,
                                      truncation=True,
                                      return_tensors='np')
        return wp_target['input_ids'][0]
