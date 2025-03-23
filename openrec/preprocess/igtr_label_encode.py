import copy
import random

import numpy as np

from openrec.preprocess.ctc_label_encode import BaseRecLabelEncode


class IGTRLabelEncode(BaseRecLabelEncode):
    """Convert between text-label and text-index."""

    def __init__(self,
                 max_text_length,
                 character_dict_path=None,
                 use_space_char=False,
                 k=1,
                 ch=False,
                 prompt_error=False,
                 **kwargs):
        super(IGTRLabelEncode,
              self).__init__(max_text_length, character_dict_path,
                             use_space_char)
        self.ignore_index = self.dict['<pad>']
        self.k = k
        self.prompt_error = prompt_error
        self.ch = ch
        rare_file = kwargs.get('rare_file', None)
        siml_file = kwargs.get('siml_file', None)
        siml_char_dict = {}
        siml_char_list = [0 for _ in range(self.num_character)]
        if siml_file is not None:
            with open(siml_file, 'r') as f:
                for lin in f.readlines():
                    lin_s = lin.strip().split('\t')
                    char_siml = lin_s[0]
                    if char_siml in self.dict:
                        siml_list = []
                        siml_prob = []
                        for i in range(1, len(lin_s), 2):
                            c = lin_s[i]
                            prob = int(lin_s[i + 1])
                            if c in self.dict and prob >= 1:
                                siml_list.append(self.dict[c])
                                siml_prob.append(prob)
                        siml_prob = np.array(siml_prob,
                                             dtype=np.float32) / sum(siml_prob)
                        siml_char_dict[self.dict[char_siml]] = [
                            siml_list, siml_prob.tolist()
                        ]
                        siml_char_list[self.dict[char_siml]] = 1
        self.siml_char_dict = siml_char_dict
        self.siml_char_list = siml_char_list

        rare_char_list = [0 for _ in range(self.num_character)]
        if rare_file is not None:
            with open(rare_file, 'r') as f:
                for lin in f.readlines():
                    lin_s = lin.strip().split('\t')
                    # print(lin_s)
                    char_rare = lin_s[0]
                    num_appear = int(lin_s[1])
                    if char_rare in self.dict and num_appear < 1000:
                        rare_char_list[self.dict[char_rare]] = 1

        self.rare_char_list = rare_char_list  # [self.dict[char] for char in rare_char_list]

    def __call__(self, data):
        text = data['label']  # coffee

        encoder_result = self.encode(text)
        if encoder_result is None:
            return None

        text, text_char_num, ques_list_s, prompt_list_s = encoder_result

        if len(text) > self.max_text_len:
            return None
        data['length'] = np.array(len(text))

        text = [self.dict['<s>']] + text + [self.dict['</s>']]
        text = text + [self.dict['<pad>']
                       ] * (self.max_text_len + 2 - len(text))
        data['label'] = np.array(text)  # 6

        ques_len_list = []
        ques2_len_list = []
        prompt_len_list = []

        prompt_pos_idx_list = []
        prompt_char_idx_list = []
        ques_pos_idx_list = []
        ques1_answer_list = []
        ques2_char_idx_list = []
        ques2_answer_list = []
        ques4_char_num_list = []
        train_step = 0
        for prompt_list, ques_list in zip(prompt_list_s, ques_list_s):

            prompt_len = len(prompt_list) + 1
            prompt_len_list.append(prompt_len)
            prompt_list = np.array(
                [[0, self.dict['<s>'], 0]] + prompt_list +
                [[self.max_text_len + 2, self.dict['<pad>'], 0]] *
                (self.max_text_len - len(prompt_list)))
            prompt_pos_idx_list.append(prompt_list[:, 0])
            prompt_char_idx_list.append(prompt_list[:, 1])

            ques_len = len(ques_list)
            ques_len_list.append(ques_len)

            ques_list = np.array(
                ques_list + [[self.max_text_len + 2, self.dict['<pad>'], 0]] *
                (self.max_text_len + 1 - ques_len))
            ques_pos_idx_list.append(ques_list[:, 0])
            # what is the first and third char?
            # Is the first character 't'? and  Is the third character 'f'?
            # How many 'c', 's' and 'f' are there in the text image?
            ques1_answer_list.append(ques_list[:, 1])
            ques2_char_idx = copy.deepcopy(ques_list[:ques_len, :2])
            new_ques2_char_idx = []
            ques2_answer = []
            for q_2, ques2_idx in enumerate(ques2_char_idx.tolist()):

                if (train_step == 2 or train_step == 3) and q_2 == ques_len - 1:
                    new_ques2_char_idx.append(ques2_idx)
                    ques2_answer.append(1)
                    continue
                if ques2_idx[1] != self.dict['<pad>'] and random.random() > 0.5:
                    select_idx = random.randint(0, self.num_character - 3)
                    new_ques2_char_idx.append([ques2_idx[0], select_idx])
                    if select_idx == ques2_idx[1]:
                        ques2_answer.append(1)
                    else:
                        ques2_answer.append(0)

                    if self.siml_char_list[
                            ques2_idx[1]] == 1 and random.random() > 0.5:
                        select_idx_sim_list = random.sample(
                            self.siml_char_dict[ques2_idx[1]][0],
                            min(3, len(self.siml_char_dict[ques2_idx[1]][0])),
                        )
                        for select_idx in select_idx_sim_list:
                            new_ques2_char_idx.append(
                                [ques2_idx[0], select_idx])
                            if select_idx == ques2_idx[1]:
                                ques2_answer.append(1)
                            else:
                                ques2_answer.append(0)
                else:
                    new_ques2_char_idx.append(ques2_idx)
                    ques2_answer.append(1)
            ques2_len_list.append(len(new_ques2_char_idx))
            ques2_char_idx_new = np.array(
                new_ques2_char_idx +
                [[self.max_text_len + 2, self.dict['<pad>']]] *
                (self.max_text_len * 4 + 1 - len(new_ques2_char_idx)))
            ques2_answer = np.array(
                ques2_answer + [0] *
                (self.max_text_len * 4 + 1 - len(ques2_answer)))
            ques2_char_idx_list.append(ques2_char_idx_new)
            ques2_answer_list.append(ques2_answer)

            ques4_char_num_list.append(ques_list[:, 2])
            train_step += 1

        data['ques_len_list'] = np.array(ques_len_list, dtype=np.int64)
        data['ques2_len_list'] = np.array(ques2_len_list, dtype=np.int64)
        data['prompt_len_list'] = np.array(prompt_len_list, dtype=np.int64)

        data['prompt_pos_idx_list'] = np.array(prompt_pos_idx_list,
                                               dtype=np.int64)
        data['prompt_char_idx_list'] = np.array(prompt_char_idx_list,
                                                dtype=np.int64)
        data['ques_pos_idx_list'] = np.array(ques_pos_idx_list, dtype=np.int64)
        data['ques1_answer_list'] = np.array(ques1_answer_list, dtype=np.int64)
        data['ques2_char_idx_list'] = np.array(ques2_char_idx_list,
                                               dtype=np.int64)
        data['ques2_answer_list'] = np.array(ques2_answer_list,
                                             dtype=np.float32)

        data['ques3_answer'] = np.array(
            text_char_num,
            dtype=np.int64)  # np.array([1, 0, 2]) # answer 1, 0, 2
        data['ques4_char_num_list'] = np.array(ques4_char_num_list)

        return data

    def add_special_char(self, dict_character):
        dict_character = ['</s>'] + dict_character + ['<s>'] + ['<pad>']
        self.num_character = len(dict_character)

        return dict_character

    def encode(self, text):
        """
        Encodes the given text into a list of character IDs and generates various lists for question and prompt sequences.

        Args:
            text (str): The input text to be encoded.

        Returns:
            tuple: A tuple containing:
                - text_list (list): A list of character IDs corresponding to the input text.
                - char_num (list): A list of character counts for each character ID.
                - ques_list (list): A list of question sequences, each sequence is a list of [position, character ID, character count].
                - prompt_list (list): A list of prompt sequences, each sequence is a list of [position, character ID, character count].

        Notes:
            - If the input text is empty, the function returns None.
            - The function handles rare and unrare characters differently.
            - The function supports both lowercased and original text based on the `self.lower` attribute.
            - The function generates additional sequences if the length of the input text is greater than 1.
        """

        if len(text) == 0:
            return None
        if self.lower:
            text = text.lower()
        char_num = [0 for _ in range(self.num_character - 2)]
        char_num[0] = 1
        text_list = []
        qa_text = []
        pos_i = 0
        rare_char_qa = []
        unrare_char_qa = []
        for char in text:
            if char not in self.dict:
                continue

            char_id = self.dict[char]
            text_list.append(char_id)
            qa_text.append([pos_i + 1, char_id, char_num[char_id]])
            if self.rare_char_list[char_id] == 1:
                rare_char_qa.append([pos_i + 1, char_id, char_num[char_id]])
            else:
                unrare_char_qa.append([pos_i + 1, char_id, char_num[char_id]])
            char_num[char_id] += 1
            pos_i += 1

        if self.ch:
            char_num_ch = []
            char_num_ch_none = []
            rare_char_num_ch_none = []
            for i, num in enumerate(char_num):
                if self.rare_char_list[i] == 1:
                    rare_char_num_ch_none.append([i, num])
                if num > 0:
                    char_num_ch.append([i, num])
                else:
                    char_num_ch_none.append([i, 0])
            none_char_index = random.sample(
                char_num_ch_none,
                min(37 - len(char_num_ch), len(char_num_ch_none)))
            if len(rare_char_num_ch_none) > 0:
                none_rare_char_index = random.sample(
                    rare_char_num_ch_none,
                    min(40 - len(char_num_ch) - len(none_char_index),
                        len(rare_char_num_ch_none)),
                )
                char_num_ch = char_num_ch + none_char_index + none_rare_char_index
            else:
                char_num_ch = char_num_ch + none_char_index
            char_num_ch.sort(key=lambda x: x[0])
            char_num = char_num_ch

        len_ = len(text_list)
        if len_ == 0:
            return None
        ques_list = [
            qa_text + [[pos_i + 1, self.dict['</s>'], 0]],
            [[pos_i + 1, self.dict['</s>'], 0]],
        ]
        prompt_list = [qa_text[len_:], qa_text]
        if len_ == 1:
            ques_list.append([[self.max_text_len + 1, self.dict['</s>'], 0]])
            prompt_list.append(
                [[self.max_text_len + 2, self.dict['<pad>'], 0]] * 4 + qa_text)
            for _ in range(1, self.k):
                ques_list.append(
                    [[self.max_text_len + 2, self.dict['<pad>'], 0]])
                prompt_list.append(qa_text[1:])
        else:

            next_id = random.sample(range(1, len_ + 1), 2)
            for slice_id in next_id:
                b_i = slice_id - 5 if slice_id - 5 > 0 else 0
                if slice_id == len_:
                    ques_list.append(
                        [[self.max_text_len + 1, self.dict['</s>'], 0]])
                else:
                    ques_list.append(
                        qa_text[slice_id:] +
                        [[self.max_text_len + 1, qa_text[slice_id][1], 0]])
                prompt_list.append(
                    [[self.max_text_len + 2, self.dict['<pad>'], 0]] *
                    (5 - slice_id + b_i) + qa_text[b_i:slice_id])

            shuffle_id1 = random.sample(range(1, len_),
                                        2) if len_ > 2 else [1, 0]
            for slice_id in shuffle_id1:
                if slice_id == 0:
                    ques_list.append(
                        [[self.max_text_len + 2, self.dict['<pad>'], 0]])
                    prompt_list.append(qa_text[:0])
                else:
                    ques_list.append(qa_text[slice_id:] +
                                     [[pos_i + 1, self.dict['</s>'], 0]])
                    prompt_list.append(qa_text[:slice_id])

            if len_ > 2:
                shuffle_id2 = random.sample(
                    range(1, len_),
                    self.k - 4 if len_ - 1 > self.k - 4 else len_ - 1)
                if self.k - 4 != len(shuffle_id2):
                    shuffle_id2 += random.sample(range(1, len_),
                                                 self.k - 4 - len(shuffle_id2))
                rare_slice_id = len(rare_char_qa)
                unrare_slice_id = len(unrare_char_qa)
                for slice_id in shuffle_id2:
                    random.shuffle(qa_text)
                    if len(rare_char_qa) > 0 and random.random() < 0.5:
                        ques_list.append(rare_char_qa[:rare_slice_id] +
                                         unrare_char_qa[unrare_slice_id:] +
                                         [[pos_i + 1, self.dict['</s>'], 0]])
                        if len(unrare_char_qa[:unrare_slice_id]) > 0:
                            prompt_list1 = random.sample(
                                unrare_char_qa[:unrare_slice_id],
                                random.randint(
                                    1, len(unrare_char_qa[:unrare_slice_id]))
                                if len(unrare_char_qa[:unrare_slice_id]) > 1
                                else 1,
                            )
                        else:
                            prompt_list1 = []
                        if len(rare_char_qa[rare_slice_id:]) > 0:
                            prompt_list2 = random.sample(
                                rare_char_qa[rare_slice_id:],
                                random.randint(
                                    1,
                                    len(rare_char_qa[rare_slice_id:])
                                    if len(rare_char_qa[rare_slice_id:]) > 1
                                    else 1,
                                ),
                            )
                        else:
                            prompt_list2 = []
                        prompt_list.append(prompt_list1 + prompt_list2)
                        random.shuffle(rare_char_qa)
                        random.shuffle(unrare_char_qa)
                        rare_slice_id = random.randint(
                            1,
                            len(rare_char_qa)) if len(rare_char_qa) > 1 else 1
                        unrare_slice_id = random.randint(
                            1, len(unrare_char_qa)
                        ) if len(unrare_char_qa) > 1 else 1
                    else:
                        ques_list.append(qa_text[slice_id:] +
                                         [[pos_i + 1, self.dict['</s>'], 0]])
                        prompt_list.append(qa_text[:slice_id])
            else:
                ques_list.append(qa_text[1:] +
                                 [[pos_i + 1, self.dict['</s>'], 0]])
                prompt_list.append(qa_text[:1])
                ques_list.append(qa_text[:1] +
                                 [[pos_i + 1, self.dict['</s>'], 0]])
                prompt_list.append(qa_text[1:])
                ques_list += [[[self.max_text_len + 2, self.dict['<pad>'], 0]]
                              ] * (self.k - 6)
                prompt_list += [qa_text[:0]] * (self.k - 6)

        return text_list, char_num, ques_list, prompt_list
