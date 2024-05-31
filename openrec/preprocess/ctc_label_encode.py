import numpy as np
import re
from random import sample
import random
from tools.utils.logging import get_logger


class BaseRecLabelEncode(object):
    """Convert between text-label and text-index"""

    def __init__(
            self,
            max_text_length,
            character_dict_path=None,
            use_space_char=False,
            lower=False, ):
        self.max_text_len = max_text_length
        self.beg_str = "sos"
        self.end_str = "eos"
        self.lower = lower
        self.reverse = False
        if character_dict_path is None:
            logger = get_logger()
            logger.warning(
                "The character_dict_path is None, model can only recognize number and lower letters"
            )
            self.character_str = "0123456789abcdefghijklmnopqrstuvwxyz"
            dict_character = list(self.character_str)
            self.lower = True
        else:
            self.character_str = []
            with open(character_dict_path, "rb") as fin:
                lines = fin.readlines()
                for line in lines:
                    line = line.decode("utf-8").strip("\n").strip("\r\n")
                    self.character_str.append(line)
            if use_space_char:
                self.character_str.append(" ")
            dict_character = list(self.character_str)
            if "arabic" in character_dict_path:
                self.reverse = True
        dict_character = self.add_special_char(dict_character)
        self.dict = {}
        for i, char in enumerate(dict_character):
            self.dict[char] = i
        self.character = dict_character

    def label_reverse(self, text):
        text_re = []
        c_current = ""
        for c in text:
            if not bool(re.search("[a-zA-Z0-9 :*./%+-١٢٣٤٥٦٧٨٩٠]", c)):
                if c_current != "":
                    text_re.append(c_current)
                text_re.append(c)
                c_current = ""
            else:
                c_current += c
        if c_current != "":
            text_re.append(c_current)

        return "".join(text_re[::-1])

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
        if len(text) == 0 or len(text) > self.max_text_len:
            return None
        if self.lower:
            text = text.lower()
        text_list = []
        for char in text:
            if char not in self.dict:
                # logger = get_logger()
                # logger.warning('{} is not in dict'.format(char))
                continue
            text_list.append(self.dict[char])
        if len(text_list) == 0:
            return None
        return text_list


class CTCLabelEncode(BaseRecLabelEncode):
    """Convert between text-label and text-index"""

    def __init__(self,
                 max_text_length,
                 character_dict_path=None,
                 use_space_char=False,
                 **kwargs):
        super(CTCLabelEncode, self).__init__(
            max_text_length, character_dict_path, use_space_char)
        self.is_reverse = kwargs.get("is_reverse", False)

    def __call__(self, data):
        text = data["label"]
        if self.reverse and self.is_reverse:  # for arabic rec
            text = self.label_reverse(text)
        text = self.encode(text)
        if text is None:
            return None
        data["length"] = np.array(len(text))
        text = text + [0] * (self.max_text_len - len(text))
        data["label"] = np.array(text)

        label = [0] * len(self.character)
        for x in text:
            label[x] += 1
        data["label_ace"] = np.array(label)
        return data

    def add_special_char(self, dict_character):
        dict_character = ["blank"] + dict_character
        return dict_character


class CPPDLabelEncode(BaseRecLabelEncode):
    """Convert between text-label and text-index"""

    def __init__(
            self,
            max_text_length,
            character_dict_path=None,
            use_space_char=False,
            ch=False,
            #  ch_7000=7000,
            ignore_index=100,
            use_sos=False,
            pos_len=False,
            **kwargs):
        self.use_sos = use_sos
        super(CPPDLabelEncode, self).__init__(
            max_text_length, character_dict_path, use_space_char)
        self.ch = ch
        self.ignore_index = ignore_index
        self.pos_len = pos_len

    def __call__(self, data):
        text = data["label"]
        if self.ch:
            # text, text_node_index, text_node_num = self.encodech(text)
            # if text is None:
            #     return None
            # if len(text) >= self.max_text_len:
            #     return None
            # data['length'] = np.array(len(text))
            # # text.insert(0, 0)
            # text_pos_node = [1] * (len(text)+1) + [0] * (self.max_text_len - len(text))

            # text.append(0)
            # ctc_text = text + [0] * (self.max_text_len - len(text))

            # text = text + [self.ignore_index] * (self.max_text_len + 1 - len(text))

            # data['label'] = np.array(text)
            # data['label_node'] = np.array(text_node_num + text_pos_node)
            # data['label_index'] = np.array(text_node_index)
            # data['label_ctc'] = np.array(ctc_text)
            # return data
            text, text_node_index, text_node_num = self.encodech(text)
            if text is None:
                return None
            if len(text) > self.max_text_len:
                return None
            data["length"] = np.array(len(text))
            # text.insert(0, 0)
            if self.pos_len:
                text_pos_node = [i_ for i_ in range(len(text), -1, -1)
                                 ] + [100] * (self.max_text_len - len(text))
            else:
                text_pos_node = [1] * (len(text) + 1) + [0] * (self.max_text_len
                                                               - len(text))

            text.append(0)
            ctc_text = text + [0] * (self.max_text_len - len(text))

            text = text + [self.ignore_index] * (self.max_text_len + 1 -
                                                 len(text))

            data["label"] = np.array(text)
            data["label_node"] = np.array(text_node_num + text_pos_node)
            data["label_index"] = np.array(text_node_index)
            # data['label_ctc'] = np.array(ctc_text)
            return data
        else:
            text, text_char_node, ch_order = self.encode(text)

            if text is None:
                return None
            if len(text) >= self.max_text_len:
                return None
            data["length"] = np.array(len(text))
            # text.insert(0, 0)
            if self.pos_len:
                text_pos_node = [i_ for i_ in range(len(text), -1, -1)
                                 ] + [100] * (self.max_text_len - len(text))
            else:
                text_pos_node = [1] * (len(text) + 1) + [0] * (self.max_text_len
                                                               - len(text))

            text.append(0)

            text = text + [self.ignore_index] * (self.max_text_len + 1 -
                                                 len(text))
            data["label"] = np.array(text)
            data["label_node"] = np.array(text_char_node + text_pos_node)
            data["label_order"] = np.array(ch_order)

            return data

    def add_special_char(self, dict_character):
        if self.use_sos:
            dict_character = ["<s>", "</s>"] + dict_character
        else:
            dict_character = ["</s>"] + dict_character
        self.num_character = len(dict_character)

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
        if len(text) == 0 or len(text) > self.max_text_len:
            return None, None
        if self.lower:
            text = text.lower()
        text_node = [0 for _ in range(self.num_character)]
        text_node[0] = 1
        text_list = []
        ch_order = []
        # ch_current = []
        # order_index = {}
        order = 1
        for char in text:
            if char not in self.dict:
                # logger = get_logger()
                # logger.warning('{} is not in dict'.format(char))
                continue
            text_list.append(self.dict[char])
            text_node[self.dict[char]] += 1
            # ch_current.append(self.dict[char])
            # if char in order:
            #     order_index[char] += 1
            # else:
            #     order_index[char] = 1

            ch_order.append(
                [self.dict[char], text_node[self.dict[char]], order])
            order += 1

        no_ch_order = []
        for char in self.character:
            if char not in text:
                no_ch_order.append([self.dict[char], 1, 0])
                # no_text_list.append(self.dict[char])
        random.shuffle(no_ch_order)
        # text_num_and_order = text_list + no_text_list
        ch_order = ch_order + no_ch_order
        ch_order = ch_order[:self.max_text_len + 1]

        if len(text_list) == 0:
            return None, None
        return text_list, text_node, ch_order.sort()

    def encodech(self, text):
        """convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]

        output:
            text: concatenated text index for CTCLoss.
                    [sum(text_lengths)] = [text_index_0 + text_index_1 + ... + text_index_(n - 1)]
            length: length of each text. [batch_size]
        """
        if len(text) == 0 or len(text) > self.max_text_len:
            return None, None, None
        if self.lower:
            text = text.lower()
        text_node_dict = {}
        # text_node = [0 for _ in range(self.num_character)]
        # text_node[0] = 1
        text_node_dict.update({0: 1})
        character_index = [_ for _ in range(self.num_character)]
        text_list = []
        for char in text:
            if char not in self.dict:
                # logger = get_logger()
                # logger.warning('{} is not in dict'.format(char))
                continue
            i_c = self.dict[char]
            text_list.append(i_c)

            # print(i_c)

            if i_c in text_node_dict.keys():
                text_node_dict[i_c] += 1
            else:
                text_node_dict.update({i_c: 1})
        for ic in list(text_node_dict.keys()):
            character_index.remove(ic)
        none_char_index = sample(character_index,
                                 37 - len(list(text_node_dict.keys())))
        for ic in none_char_index:
            text_node_dict[ic] = 0

        text_node_index = sorted(text_node_dict)
        # text_node_index = list(text_node_dict.keys())

        text_node_num = [text_node_dict[k] for k in text_node_index
                         ]  # list(text_node_dict.values())
        if len(text_list) == 0:
            return None, None, None
        return text_list, text_node_index, text_node_num

    # def encodech(self, text):
    #     """convert text-label into text-index.
    #     input:
    #         text: text labels of each image. [batch_size]

    #     output:
    #         text: concatenated text index for CTCLoss.
    #                 [sum(text_lengths)] = [text_index_0 + text_index_1 + ... + text_index_(n - 1)]
    #         length: length of each text. [batch_size]
    #     """
    #     if len(text) == 0 or len(text) > self.max_text_len:
    #         return None, None, None
    #     if self.lower:
    #         text = text.lower()
    #     text_node_dict = {}
    #     # text_node = [0 for _ in range(self.num_character)]
    #     # text_node[0] = 1
    #     text_node_dict.update({0:1})
    #     text_list = []
    #     for char in text:
    #         if char not in self.dict:
    #             # logger = get_logger()
    #             # logger.warning('{} is not in dict'.format(char))
    #             continue
    #         i_c = self.dict[char]
    #         text_list.append(i_c)

    #         if i_c in text_node_dict.keys():
    #             text_node_dict[i_c] += 1
    #         else:
    #             text_node_dict.update({i_c:1})
    #     # text_node_dict = sorted(text_node_dict)
    #     text_node_index = list(text_node_dict.keys())
    #     text_node_num = list(text_node_dict.values())
    #     if len(text_list) == 0:
    #         return None, None, None
    #     return text_list, text_node_index + [-1] * (self.max_text_len - len(text_node_index)), text_node_num + [self.ignore_index] * (self.max_text_len - len(text_node_num))
