import numpy as np
import torch

from .nrtr_postprocess import NRTRLabelDecode


class IGTRLabelDecode(NRTRLabelDecode):
    """Convert between text-label and text-index."""

    def __init__(self,
                 character_dict_path=None,
                 use_space_char=False,
                 **kwargs):
        super(IGTRLabelDecode, self).__init__(character_dict_path,
                                              use_space_char)

    def __call__(self, preds, batch=None, *args, **kwargs):

        if isinstance(preds, list):
            if isinstance(preds[0], dict):
                preds = preds[-1].detach().cpu().numpy()
                if isinstance(preds, torch.Tensor):
                    preds = preds.detach().cpu().numpy()
                elif isinstance(preds, dict):
                    preds = preds['align'][-1].detach().cpu().numpy()
                else:
                    preds = preds
                preds_idx = preds.argmax(axis=2)
                preds_prob = preds.max(axis=2)
                text = self.decode(preds_idx,
                                   preds_prob,
                                   is_remove_duplicate=False)
            else:
                preds_idx = preds[0].detach().cpu().numpy()
                preds_prob = preds[1].detach().cpu().numpy()
                text = self.decode(preds_idx,
                                   preds_prob,
                                   is_remove_duplicate=False)
        else:
            if isinstance(preds, torch.Tensor):
                preds = preds.detach().cpu().numpy()
            elif isinstance(preds, dict):
                preds = preds['align'][-1].detach().cpu().numpy()
            else:
                preds = preds
            preds_idx = preds.argmax(axis=2)
            preds_idx_top5 = preds.argsort(axis=2)[:, :, -5:]
            preds_prob = preds.max(axis=2)
            text = self.decode(preds_idx,
                               preds_prob,
                               is_remove_duplicate=False,
                               idx_top5=preds_idx_top5)
        if batch is None:
            return text
        label = batch[1]
        label = self.decode(label)
        return text, label

    def add_special_char(self, dict_character):
        dict_character = ['</s>'] + dict_character + ['<s>', '<pad>']
        return dict_character

    def decode(self,
               text_index,
               text_prob=None,
               is_remove_duplicate=False,
               idx_top5=None):
        """convert text-index into text-label."""
        result_list = []
        batch_size = len(text_index)
        for batch_idx in range(batch_size):
            char_list = []
            char_list_top5 = []
            conf_list = []
            for idx in range(len(text_index[batch_idx])):
                char_idx_top5 = []
                try:
                    char_idx = self.character[int(text_index[batch_idx][idx])]
                    if idx_top5 is not None:
                        for top5_i in idx_top5[batch_idx][idx]:
                            char_idx_top5.append(self.character[top5_i])
                except:
                    continue
                if char_idx == '</s>':  # end
                    break
                if char_idx == '<s>' or char_idx == '<pad>':
                    continue
                char_list.append(char_idx)
                char_list_top5.append(char_idx_top5)
                if text_prob is not None:
                    conf_list.append(text_prob[batch_idx][idx])
                else:
                    conf_list.append(1)
            text = ''.join(char_list)
            if idx_top5 is not None:
                result_list.append(
                    (text, [np.mean(conf_list).tolist(), char_list_top5]))
            else:
                result_list.append((text, np.mean(conf_list).tolist()))
        return result_list
