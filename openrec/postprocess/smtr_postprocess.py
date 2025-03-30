import numpy as np
import torch

from .ctc_postprocess import BaseRecLabelDecode


class SMTRLabelDecode(BaseRecLabelDecode):
    """Convert between text-label and text-index."""

    BOS = '<s>'
    EOS = '</s>'
    IN_F = '<INF>'  # ignore
    IN_B = '<INB>'  # ignore
    PAD = '<pad>'

    def __init__(self,
                 character_dict_path=None,
                 use_space_char=True,
                 next_mode=True,
                 **kwargs):
        super(SMTRLabelDecode, self).__init__(character_dict_path,
                                              use_space_char)
        self.next_mode = next_mode

    def __call__(self, preds, batch=None, *args, **kwargs):
        if isinstance(preds, list):
            preds = preds[-1]
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()
        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)
        text = self.decode(preds_idx, preds_prob, is_remove_duplicate=False)
        if batch is None:
            return text
        label = batch[1]
        label = self.decode(label[:, 1:])
        return text, label

    def add_special_char(self, dict_character):
        dict_character = [self.EOS] + dict_character + [
            self.BOS, self.IN_F, self.IN_B, self.PAD
        ]
        self.num_character = len(dict_character)
        return dict_character

    def decode(self, text_index, text_prob=None, is_remove_duplicate=False):
        """convert text-index into text-label."""
        result_list = []
        batch_size = len(text_index)
        for batch_idx in range(batch_size):
            char_list = []
            conf_list = []
            for idx in range(len(text_index[batch_idx])):
                try:
                    char_idx = self.character[int(text_index[batch_idx][idx])]
                except:
                    continue
                if char_idx == '</s>':  # end
                    break
                if char_idx == '<s>' or char_idx == '<pad>':
                    continue
                char_list.append(char_idx)

                if text_prob is not None:
                    conf_list.append(text_prob[batch_idx][idx])
                else:
                    conf_list.append(1)
            if self.next_mode or text_prob is None:
                text = ''.join(char_list)
            else:
                text = ''.join(char_list[::-1])
            result_list.append((text, np.mean(conf_list).tolist()))
        return result_list
