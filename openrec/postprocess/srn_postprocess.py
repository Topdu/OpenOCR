import numpy as np
import torch

from .ctc_postprocess import BaseRecLabelDecode


class SRNLabelDecode(BaseRecLabelDecode):
    """Convert between text-label and text-index."""

    def __init__(self,
                 character_dict_path=None,
                 use_space_char=False,
                 **kwargs):
        super(SRNLabelDecode, self).__init__(character_dict_path,
                                             use_space_char)
        self.max_len = 25

    def add_special_char(self, dict_character):
        dict_character = dict_character + ['<BOS>', '<EOS>']
        self.start_idx = len(dict_character) - 2
        self.end_idx = len(dict_character) - 1
        return dict_character

    def decode(self, text_index, text_prob=None, is_remove_duplicate=False):
        """convert text-index into text-label."""
        result_list = []
        ignored_tokens = self.get_ignored_tokens()
        # [B,25]
        batch_size = len(text_index)
        for batch_idx in range(batch_size):
            char_list = []
            conf_list = []
            for idx in range(len(text_index[batch_idx])):
                # print(f"text_index[{batch_idx}][{idx}]:{text_index[batch_idx][idx]}")
                if text_index[batch_idx][idx] in ignored_tokens:
                    continue
                if int(text_index[batch_idx][idx]) == int(self.end_idx):
                    if text_prob is None and idx == 0:
                        continue
                    else:
                        break
                if is_remove_duplicate:
                    # only for predict
                    if idx > 0 and text_index[batch_idx][
                            idx - 1] == text_index[batch_idx][idx]:
                        continue
                char_list.append(self.character[int(
                    text_index[batch_idx][idx])])
                if text_prob is not None:
                    conf_list.append(text_prob[batch_idx][idx])
                else:
                    conf_list.append(1)
            text = ''.join(char_list)
            result_list.append((text, np.mean(conf_list).tolist()))
        return result_list

    def __call__(self, preds, batch=None, *args, **kwargs):

        if isinstance(preds, torch.Tensor):
            preds = preds.reshape([-1, self.max_len, preds.shape[-1]])
            preds = preds.detach().cpu().numpy()
        else:
            preds = preds[-1]
            preds = preds.reshape([-1, self.max_len,
                                   preds.shape[-1]]).detach().cpu().numpy()

        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)
        text = self.decode(preds_idx, preds_prob, is_remove_duplicate=False)

        if batch is None:
            return text

        label = batch[1]
        # print(f"label.shape:{label.shape}")
        label = self.decode(label, is_remove_duplicate=False)
        return text, label

    def get_ignored_tokens(self):
        return [self.start_idx, self.end_idx]
