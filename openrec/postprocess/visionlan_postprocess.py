import numpy as np
import torch
import torch.nn.functional as F

from .ctc_postprocess import BaseRecLabelDecode


class VisionLANLabelDecode(BaseRecLabelDecode):
    """Convert between text-label and text-index."""

    def __init__(self,
                 character_dict_path=None,
                 use_space_char=False,
                 **kwargs):
        super(VisionLANLabelDecode, self).__init__(character_dict_path,
                                                   use_space_char)
        self.max_text_length = kwargs.get('max_text_length', 25)
        self.nclass = len(self.character) + 1

    def decode(self, text_index, text_prob=None, is_remove_duplicate=False):
        """convert text-index into text-label."""
        result_list = []
        ignored_tokens = self.get_ignored_tokens()
        batch_size = len(text_index)
        for batch_idx in range(batch_size):
            selection = np.ones(len(text_index[batch_idx]), dtype=bool)
            if is_remove_duplicate:
                selection[1:] = text_index[batch_idx][1:] != text_index[
                    batch_idx][:-1]
            for ignored_token in ignored_tokens:
                selection &= text_index[batch_idx] != ignored_token

            char_list = [
                self.character[text_id - 1]
                for text_id in text_index[batch_idx][selection]
            ]
            if text_prob is not None:
                conf_list = text_prob[batch_idx][selection]
            else:
                conf_list = [1] * len(selection)
            if len(conf_list) == 0:
                conf_list = [0]

            text = ''.join(char_list)
            result_list.append((text, np.mean(conf_list).tolist()))
        return result_list

    def __call__(self, preds, batch=None, *args, **kwargs):
        if len(preds) == 2:  # eval mode
            net_out, length = preds
            if batch is not None:
                label = batch[1]

        else:  # train mode
            net_out = preds[0]
            label, length = batch[1], batch[5]
            net_out = torch.cat([t[:l] for t, l in zip(net_out, length)],
                                dim=0)
        text = []
        if not isinstance(net_out, torch.Tensor):
            net_out = torch.tensor(net_out, dtype=torch.float32)
        net_out = F.softmax(net_out, dim=1)
        for i in range(0, length.shape[0]):
            preds_idx = (net_out[int(length[:i].sum()):int(length[:i].sum() +
                                                           length[i])].topk(1)
                         [1][:, 0].tolist())
            preds_text = ''.join([
                self.character[idx - 1]
                if idx > 0 and idx <= len(self.character) else ''
                for idx in preds_idx
            ])
            preds_prob = net_out[int(length[:i].sum()):int(length[:i].sum() +
                                                           length[i])].topk(
                                                               1)[0][:, 0]
            preds_prob = torch.exp(
                torch.log(preds_prob).sum() / (preds_prob.shape[0] + 1e-6))
            text.append((preds_text, float(preds_prob)))
        if batch is None:
            return text
        label = self.decode(label)
        return text, label
