import numpy as np
import torch

from .ctc_postprocess import BaseRecLabelDecode


class NRTRLabelDecode(BaseRecLabelDecode):
    """Convert between text-label and text-index."""

    def __init__(self,
                 character_dict_path=None,
                 use_space_char=True,
                 **kwargs):
        super(NRTRLabelDecode, self).__init__(character_dict_path,
                                              use_space_char)

    def __call__(self, preds, batch=None, *args, **kwargs):
        preds = preds['res']
        if len(preds) == 2:
            preds_id = preds[0]
            preds_prob = preds[1]
            if isinstance(preds_id, torch.Tensor):
                preds_id = preds_id.detach().cpu().numpy()
            if isinstance(preds_prob, torch.Tensor):
                preds_prob = preds_prob.detach().cpu().numpy()
            if preds_id[0][0] == 2:
                preds_idx = preds_id[:, 1:]
                preds_prob = preds_prob[:, 1:]
            else:
                preds_idx = preds_id
            text = self.decode(preds_idx,
                               preds_prob,
                               is_remove_duplicate=False)
            if batch is None:
                return text
            label = self.decode(batch[1][:, 1:])
        else:
            if isinstance(preds, torch.Tensor):
                preds = preds.detach().cpu().numpy()
            preds_idx = preds.argmax(axis=2)
            preds_prob = preds.max(axis=2)
            text = self.decode(preds_idx,
                               preds_prob,
                               is_remove_duplicate=False)
            if batch is None:
                return text
            label = self.decode(batch[1][:, 1:])
        return text, label

    def add_special_char(self, dict_character):
        dict_character = ['blank', '<unk>', '<s>', '</s>'] + dict_character
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
                char_list.append(char_idx)
                if text_prob is not None:
                    conf_list.append(text_prob[batch_idx][idx])
                else:
                    conf_list.append(1)
            text = ''.join(char_list)
            result_list.append((text, np.mean(conf_list).tolist()))
        return result_list
