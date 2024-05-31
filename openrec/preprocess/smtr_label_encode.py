import copy
import random

import numpy as np

from openrec.preprocess.ctc_label_encode import BaseRecLabelEncode


class SMTRLabelEncode(BaseRecLabelEncode):
    """Convert between text-label and text-index."""

    BOS = '<s>'
    EOS = '</s>'
    IN_F = '<INF>'  # ignore
    IN_B = '<INB>'  # ignore
    PAD = '<pad>'

    def __init__(self,
                 max_text_length,
                 character_dict_path=None,
                 use_space_char=False,
                 sub_str_len=5,
                 **kwargs):

        super(SMTRLabelEncode,
              self).__init__(max_text_length, character_dict_path,
                             use_space_char)
        self.substr_len = sub_str_len
        self.rang_subs = [i for i in range(1, self.substr_len + 1)]
        self.idx_char = [i for i in range(1, self.num_character - 5)]

    def __call__(self, data):
        text = data['label']
        text = self.encode(text)
        if text is None:
            return None
        if len(text) > self.max_text_len:
            return None

        data['length'] = np.array(len(text))
        text_in = [self.dict[self.IN_F]] * (self.substr_len) + text + [
            self.dict[self.IN_B]
        ] * (self.substr_len)

        sub_string_list_pre = []
        next_label_pre = []
        sub_string_list = []
        next_label = []
        for i in range(self.substr_len, len(text_in) - self.substr_len):

            sub_string_list.append(text_in[i - self.substr_len:i])
            next_label.append(text_in[i])

            if self.substr_len - i == 0:
                sub_string_list_pre.append(text_in[-i:])
            else:
                sub_string_list_pre.append(text_in[-i:self.substr_len - i])

            next_label_pre.append(text_in[-(i + 1)])

        sub_string_list.append(
            [self.dict[self.IN_F]] *
            (self.substr_len - len(text[-self.substr_len:])) +
            text[-self.substr_len:])
        next_label.append(self.dict[self.EOS])
        sub_string_list_pre.append(
            text[:self.substr_len] + [self.dict[self.IN_B]] *
            (self.substr_len - len(text[:self.substr_len])))
        next_label_pre.append(self.dict[self.EOS])

        for sstr, l in zip(sub_string_list[self.substr_len:],
                           next_label[self.substr_len:]):

            id_shu = np.random.choice(self.rang_subs, 2)

            sstr1 = copy.deepcopy(sstr)
            sstr1[id_shu[0] - 1] = random.randint(1, self.num_character - 5)
            if sstr1 not in sub_string_list:
                sub_string_list.append(sstr1)
                next_label.append(l)

            sstr[id_shu[1] - 1] = random.randint(1, self.num_character - 5)

        for sstr, l in zip(sub_string_list_pre[self.substr_len:],
                           next_label_pre[self.substr_len:]):

            id_shu = np.random.choice(self.rang_subs, 2)

            sstr1 = copy.deepcopy(sstr)
            sstr1[id_shu[0] - 1] = random.randint(1, self.num_character - 5)
            if sstr1 not in sub_string_list_pre:
                sub_string_list_pre.append(sstr1)
                next_label_pre.append(l)
            sstr[id_shu[1] - 1] = random.randint(1, self.num_character - 5)

        data['length_subs'] = np.array(len(sub_string_list))
        sub_string_list = sub_string_list + [
            [self.dict[self.PAD]] * self.substr_len
        ] * ((self.max_text_len * 2) + 2 - len(sub_string_list))
        next_label = next_label + [self.dict[self.PAD]] * (
            (self.max_text_len * 2) + 2 - len(next_label))
        data['label_subs'] = np.array(sub_string_list)
        data['label_next'] = np.array(next_label)

        data['length_subs_pre'] = np.array(len(sub_string_list_pre))
        sub_string_list_pre = sub_string_list_pre + [
            [self.dict[self.PAD]] * self.substr_len
        ] * ((self.max_text_len * 2) + 2 - len(sub_string_list_pre))
        next_label_pre = next_label_pre + [self.dict[self.PAD]] * (
            (self.max_text_len * 2) + 2 - len(next_label_pre))
        data['label_subs_pre'] = np.array(sub_string_list_pre)
        data['label_next_pre'] = np.array(next_label_pre)

        text = [self.dict[self.BOS]] + text + [self.dict[self.EOS]]
        text = text + [self.dict[self.PAD]
                       ] * (self.max_text_len + 2 - len(text))
        data['label'] = np.array(text)
        return data

    def add_special_char(self, dict_character):
        dict_character = [self.EOS] + dict_character + [
            self.BOS, self.IN_F, self.IN_B, self.PAD
        ]
        self.num_character = len(dict_character)
        return dict_character
