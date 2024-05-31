import numpy as np
import torch

from .ctc_postprocess import BaseRecLabelDecode


class CharLabelDecode(BaseRecLabelDecode):
    """Convert between text-label and text-index."""

    def __init__(self,
                 character_dict_path=None,
                 use_space_char=True,
                 **kwargs):
        super(CharLabelDecode, self).__init__(character_dict_path,
                                              use_space_char)

    def __call__(self, preds, label=None, *args, **kwargs):
        if len(preds) >= 4:
            preds_id = preds[0]
            preds_prob = preds[1]
            char_preds = preds[2]
            if isinstance(preds_id, torch.Tensor):
                preds_id = preds_id.numpy()
            if isinstance(preds_prob, torch.Tensor):
                preds_prob = preds_prob.numpy()
            if preds_id[0][0] == 2:
                preds_idx = preds_id[:, 1:]
                preds_prob = preds_prob[:, 1:]
                # char_preds = char_preds[:, 1:]
            else:
                preds_idx = preds_id
            char_preds = char_preds.numpy()
            char_preds_idx = char_preds.argmax(-1) + 4
            char_preds_prob = char_preds.max(-1)
            text, text_box = self.decode(preds_idx, preds_prob, char_preds_idx,
                                         char_preds_prob)
        else:
            preds_logit = preds[0].numpy()
            char_preds = preds[1].numpy()
            # if isinstance(preds, torch.Tensor):
            #     preds = preds.numpy()
            preds_idx = preds_logit.argmax(axis=2)
            preds_prob = preds_logit.max(axis=2)
            char_preds_idx = char_preds.argmax(-1) + 4
            char_preds_prob = char_preds.max(-1)
            text, text_box = self.decode(preds_idx, preds_prob, char_preds_idx,
                                         char_preds_prob)

        if label is None:
            return text, text_box
        label = self.decode(label[:, 1:])
        return text, text_box, label

    def add_special_char(self, dict_character):
        dict_character = ['blank', '<unk>', '<s>', '</s>'] + dict_character
        return dict_character

    def decode(
        self,
        text_index,
        text_prob=None,
        char_text_index=None,
        char_text_prob=None,
        is_remove_duplicate=False,
    ):
        """convert text-index into text-label."""
        result_list = []
        box_result_list = []
        batch_size = len(text_index)
        for batch_idx in range(batch_size):
            char_list = []
            conf_list = []
            char_box_list = []
            conf_box_list = []
            for idx in range(len(text_index[batch_idx])):
                try:
                    char_idx = self.character[int(text_index[batch_idx][idx])]
                    if char_text_index is not None:
                        char_box_idx = self.character[int(
                            char_text_index[batch_idx][idx])]
                except:
                    continue
                if char_idx == '</s>':  # end
                    break
                char_list.append(char_idx)

                if char_text_index is not None:
                    char_box_list.append(char_box_idx)

                if text_prob is not None:
                    conf_list.append(text_prob[batch_idx][idx])
                else:
                    conf_list.append(1)

                if char_text_prob is not None:
                    conf_box_list.append(char_text_prob[batch_idx][idx])
                else:
                    conf_box_list.append(1)
            text = ''.join(char_list)
            result_list.append((text, np.mean(conf_list).tolist()))

            if char_text_index is not None:
                text_box = ''.join(char_box_list)
                box_result_list.append(
                    (text_box, np.mean(conf_box_list).tolist()))
        if char_text_index is not None:
            return result_list, box_result_list
        return result_list
