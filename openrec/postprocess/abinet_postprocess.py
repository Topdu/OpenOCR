import torch

from .nrtr_postprocess import NRTRLabelDecode


class ABINetLabelDecode(NRTRLabelDecode):
    """Convert between text-label and text-index."""

    def __init__(self,
                 character_dict_path=None,
                 use_space_char=False,
                 **kwargs):
        super(ABINetLabelDecode, self).__init__(character_dict_path,
                                                use_space_char)

    def __call__(self, preds, batch=None, *args, **kwargs):
        if isinstance(preds, dict):
            if len(preds['align']) > 0:
                preds = preds['align'][-1].detach().cpu().numpy()
            else:
                preds = preds['vision'].detach().cpu().numpy()
        elif isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()
        else:
            preds = preds

        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)
        text = self.decode(preds_idx, preds_prob, is_remove_duplicate=False)
        if batch is None:
            return text
        label = self.decode(batch[1])
        return text, label

    def add_special_char(self, dict_character):
        dict_character = ['</s>'] + dict_character
        return dict_character
