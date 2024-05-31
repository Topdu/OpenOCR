import paddle
from .nrtr_postprocess import NRTRLabelDecode


class CPPDLabelDecode(NRTRLabelDecode):
    """ Convert between text-label and text-index """

    def __init__(self, character_dict_path=None, use_space_char=False,
                 **kwargs):
        super(CPPDLabelDecode, self).__init__(character_dict_path,
                                              use_space_char)

    def __call__(self, preds, batch=None, *args, **kwargs):
        label = batch[1]
        if isinstance(preds, tuple) or isinstance(preds, list):
            if isinstance(preds[-1], dict):
                preds = preds[-1]['align'][-1].numpy()
            else:
                preds = preds[-1].numpy()
        if isinstance(preds, paddle.Tensor):
            preds = preds.numpy()
        else:
            preds = preds
        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)
        text = self.decode(preds_idx, preds_prob, is_remove_duplicate=False)
        if label is None:
            return text
        label = self.decode(label)
        return text, label

    def add_special_char(self, dict_character):
        dict_character = ['</s>'] + dict_character
        return dict_character
