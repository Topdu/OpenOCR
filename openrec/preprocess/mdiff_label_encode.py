import random
import numpy as np

from openrec.preprocess.ctc_label_encode import BaseRecLabelEncode


class MDiffLabelEncode(BaseRecLabelEncode):
    """Convert between text-label and text-index."""

    MASK = '<mask>'
    EOS = '</s>'
    PAD = '<pad>'

    def __init__(self,
                 max_text_length,
                 character_dict_path=None,
                 use_space_char=False,
                 semi_ar=False,
                 mask_tpye=[0, 1, 2, 3, 4, 5],
                 train_all_layer=False,
                 sample_num=1,
                 **kwargs):
        super(MDiffLabelEncode,
              self).__init__(max_text_length, character_dict_path,
                             use_space_char)

        self.semi_ar = semi_ar
        self.mask_tpye = mask_tpye  #[0, 1, 2, 3, 4, 5]
        # 5种mask：
        # 全部mask，纯并行
        # 正向自回归
        # 反向自回归
        # block mask（1, 2，3，4，5，6步）
        # 随机mask
        self.train_all_layer = train_all_layer
        if not self.train_all_layer:
            self.sample_num = 1
        else:
            self.sample_num = sample_num

    def random_mask(self, text):
        l = len(text)
        p_mask = random.random()

        noisy_batch = text[:]
        masked_indices = [False] * l
        none_pad_indices = []
        for i in range(l):
            if random.random() < p_mask and text[i] != self.dict[self.PAD]:
                noisy_batch[i] = self.dict[self.MASK]
                masked_indices[i] = True
            if text[i] != self.dict[self.PAD]:
                none_pad_indices.append(i)
            if noisy_batch[i] == self.dict[self.PAD]:
                noisy_batch[i] = self.dict[self.MASK]

        if not any(masked_indices) and len(none_pad_indices) > 0:
            idx = random.choice(none_pad_indices)
            noisy_batch[idx] = self.dict[self.MASK]
            masked_indices[idx] = True
        return noisy_batch, masked_indices

    def full_mask(self, text):
        noisy_batch = [self.dict[self.MASK]] * (self.max_text_len + 1)
        masked_indices = [True] * len(text) + [False] * (self.max_text_len +
                                                         1 - len(text))
        return noisy_batch, masked_indices

    def left_to_right_mask(self, text):
        rand_split = random.randint(1, len(text) - 1)
        noisy_batch = text[:rand_split] + [self.dict[self.MASK]] * (
            self.max_text_len + 1 - rand_split)
        masked_indices = [False] * rand_split + [True] * (len(
            text) - rand_split) + [False] * (self.max_text_len + 1 - len(text))
        return noisy_batch, masked_indices

    def semi_left_to_right_mask(self, text, block_size=5, step=3, i=0):

        text = text + [self.dict[self.PAD]
                       ] * (self.max_text_len + 1 - len(text))
        noisy_batch_step_i = []
        masked_indices_step_i = []

        if i == 0:
            for t in text[:block_size]:
                if t == self.dict[self.PAD]:
                    masked_indices_step_i.append(False)
                else:
                    masked_indices_step_i.append(True)

            noisy_batch_step_i = [self.dict[self.MASK]] * block_size + [
                self.dict[self.PAD]
            ] * (self.max_text_len + 1)
            masked_indices_step_i = masked_indices_step_i + [False] * (
                self.max_text_len + 1 - block_size)
        elif i == 1:

            noisy_batch_r, masked_indices_r = self.random_mask(
                text[:block_size])
            noisy_batch_step_i = noisy_batch_r
            masked_indices_step_i = masked_indices_r
            if i == step - 1:
                for t in text[i * block_size:]:
                    if t == self.dict[self.PAD]:
                        masked_indices_step_i.append(False)
                    else:
                        masked_indices_step_i.append(True)
                    noisy_batch_step_i.append(self.dict[self.MASK])
            else:
                for t in text[i * block_size:(i + 1) * block_size]:
                    if t == self.dict[self.PAD]:
                        masked_indices_step_i.append(False)
                    else:
                        masked_indices_step_i.append(True)
                    noisy_batch_step_i.append(self.dict[self.MASK])

            noisy_batch_step_i = noisy_batch_step_i + [self.dict[self.PAD]] * (
                self.max_text_len + 1 - len(noisy_batch_step_i))
            masked_indices_step_i = masked_indices_step_i + [False] * (
                self.max_text_len + 1 - len(noisy_batch_step_i))
        elif i >= 2:

            for t in text[:(i - 1) * block_size]:
                if t == self.dict[self.PAD]:
                    noisy_batch_step_i.append(self.dict[self.MASK])
                else:
                    noisy_batch_step_i.append(text[t])
            masked_indices_step_i = [False] * ((i - 1) * block_size)

            noisy_batch_r, masked_indices_r = self.random_mask(
                text[(i - 1) * block_size:i * block_size])

            noisy_batch_step_i = noisy_batch_step_i + noisy_batch_r
            masked_indices_step_i = masked_indices_step_i + masked_indices_r
            if i == step - 1:
                for t in text[i * block_size:]:
                    if t == self.dict[self.PAD]:
                        masked_indices_step_i.append(False)
                    else:
                        masked_indices_step_i.append(True)
                    noisy_batch_step_i.append(self.dict[self.MASK])
            else:
                for t in text[i * block_size:(i + 1) * block_size]:
                    if t == self.dict[self.PAD]:
                        masked_indices_step_i.append(False)
                    else:
                        masked_indices_step_i.append(True)
                    noisy_batch_step_i.append(self.dict[self.MASK])

            noisy_batch_step_i = noisy_batch_step_i + [self.dict[self.PAD]] * (
                self.max_text_len + 1 - len(noisy_batch_step_i))
            masked_indices_step_i = masked_indices_step_i + [False] * (
                self.max_text_len + 1 - len(masked_indices_step_i))
        return noisy_batch_step_i, masked_indices_step_i

    def forward_process_semi_ar(self, text):

        step = 5
        block_size = (self.max_text_len + 1) // step
        noisy_batch_semi_ar = []
        masked_indices_semi_ar = []
        for i in range(step):
            noisy_batch_step_i, masked_indices_step_i = self.semi_left_to_right_mask(
                text, block_size=block_size, step=step, i=i)
            noisy_batch_semi_ar.append(noisy_batch_step_i)
            masked_indices_semi_ar.append(masked_indices_step_i)

        rd_step = random.choice([2, 3, 4, 6, 7, 8])
        block_size = (self.max_text_len + 1) // rd_step
        rd_step_i = random.randint(0, rd_step - 1)
        noisy_batch, masked_indices = self.semi_left_to_right_mask(
            text, block_size=block_size, step=rd_step, i=rd_step_i)
        noisy_batch_semi_ar.append(noisy_batch)
        masked_indices_semi_ar.append(masked_indices)

        # 随机将text中的部分token mask掉
        noisy_batch, masked_indices = self.random_mask(text)
        noisy_batch = noisy_batch + [self.dict[self.MASK]
                                     ] * (self.max_text_len + 1 - len(text))
        masked_indices = masked_indices + [False] * (self.max_text_len + 1 -
                                                     len(text))

        noisy_batch_semi_ar.append(noisy_batch)
        masked_indices_semi_ar.append(masked_indices)
        return noisy_batch_semi_ar, noisy_batch_semi_ar

    def right_to_left_mask(self, text):
        rand_split = random.randint(1, len(text) - 1)
        noisy_batch = [self.dict[self.MASK]
                       ] * rand_split + text[rand_split:] + [
                           self.dict[self.MASK]
                       ] * (self.max_text_len + 1 - len(text))
        masked_indices = [True] * rand_split + [False] * (len(
            text) - rand_split) + [False] * (self.max_text_len + 1 - len(text))
        return noisy_batch, masked_indices

    def forward_process(self, text):

        rand_choice = random.choice(self.mask_tpye)
        if rand_choice == 0:  # 并行mask full mask
            return self.full_mask(text)
        elif rand_choice == 1 and len(text) > 2:  # 正向自回归 right mask
            return self.left_to_right_mask(text)
        elif rand_choice == 2 and len(text) > 2:  # 反向自回归 left mask
            return self.right_to_left_mask(text)
        elif rand_choice == 3 and len(text) > 2:  # block mask
            rand_step = min(random.randint(2, 6), len(text))
            if rand_step <= 1:  # len(text) <= 1
                return self.full_mask(text)
            block_size = len(text) // rand_step
            if block_size == 1:
                return self.left_to_right_mask(text) if random.random(
                ) < 0.5 else self.right_to_left_mask(text)
            # 余数处理
            if len(text) % rand_step != 0:
                rand_step += 1
            # 选择一个随机的block_size
            rand_step_from_mask = random.randint(2, rand_step)
            if rand_step == 2:
                rand_step_from_mask = 1
            else:
                rand_step_from_mask = random.randint(2, rand_step)

            noisy_batch = text[:block_size * (rand_step_from_mask - 1)]
            masked_indices = [False] * (block_size * (rand_step_from_mask - 1))

            noisy_batch = noisy_batch + [self.dict[self.MASK]] * (
                self.max_text_len + 1 - len(noisy_batch))
            masked_indices = masked_indices + [True] * (
                len(text) - block_size *
                (rand_step_from_mask - 1)) + [False] * (self.max_text_len + 1 -
                                                        len(text))
            return noisy_batch, masked_indices
        elif rand_choice == 4 and len(text) > 2:  # cloze mask
            noisy_batch = text[:]
            masked_indices = [False] * len(text)
            rand_index = random.randint(0, len(text) - 1)
            noisy_batch[rand_index] = self.dict[self.MASK]
            masked_indices[rand_index] = True
            noisy_batch = noisy_batch + [self.dict[self.MASK]] * (
                self.max_text_len + 1 - len(text))
            masked_indices = masked_indices + [False] * (self.max_text_len +
                                                         1 - len(text))
            return noisy_batch, masked_indices
        else:  # random mask
            # 随机将text中的部分token mask掉
            noisy_batch, masked_indices = self.random_mask(text)
            noisy_batch = noisy_batch + [self.dict[self.MASK]] * (
                self.max_text_len + 1 - len(text))
            masked_indices = masked_indices + [False] * (self.max_text_len +
                                                         1 - len(text))
            return noisy_batch, masked_indices

    def reflect_random_idices(self, text, eps=1e-3):
        l = len(text)
        t = random.random()
        p_mask = (1 - eps) * t + eps
        reflect_ids = text[:]
        for i in range(l):
            if random.random() < p_mask:
                reflect_ids[i] = random.randint(0, len(self.dict) - 1)
        return reflect_ids

    def __call__(self, data):
        text = data['label']
        text = self.encode(text)
        if text is None:
            return None
        data['length'] = np.array(len(text) + 1)
        text = text + [self.dict[self.EOS]]
        p_mask_list = []
        noisy_batch_list = []
        masked_indices_list = []
        reflect_ids_list = []
        for i in range(self.sample_num):
            reflect_ids = self.reflect_random_idices(text)
            reflect_ids = reflect_ids + [self.dict[self.MASK]] * (
                self.max_text_len + 1 - len(reflect_ids))
            if self.semi_ar:
                noisy_batch, masked_indices = self.forward_process_semi_ar(
                    text)
            else:
                noisy_batch, masked_indices = self.forward_process(text)
                p_mask = float(sum(masked_indices)) / float(len(text))
                p_mask_list.append(np.array(p_mask))
            noisy_batch_list.append(np.array(noisy_batch))
            masked_indices_list.append(np.array(masked_indices))
            reflect_ids_list.append(np.array(reflect_ids))

        if not self.semi_ar:
            data['p_mask'] = np.array(
                p_mask_list) if self.train_all_layer else np.array(
                    p_mask_list[0])
        data['noisy_batch'] = np.array(
            noisy_batch_list) if self.train_all_layer else np.array(
                noisy_batch_list[0])
        data['masked_indices'] = np.array(
            masked_indices_list) if self.train_all_layer else np.array(
                masked_indices_list[0])
        data['reflect_ids'] = np.array(
            reflect_ids_list) if self.train_all_layer else np.array(
                reflect_ids_list[0])

        text = text + [self.dict[self.PAD]
                       ] * (self.max_text_len + 1 - len(text))
        data['label'] = np.array(text)

        return data

    def add_special_char(self, dict_character):
        dict_character = [self.EOS] + dict_character + [self.MASK, self.PAD]
        return dict_character
