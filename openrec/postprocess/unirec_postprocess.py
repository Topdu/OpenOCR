from .ctc_postprocess import BaseRecLabelDecode


class UniRecLabelDecode(BaseRecLabelDecode):
    """Convert between text-label and text-index."""
    SPACE = '[s]'
    GO = '[GO]'
    list_token = [GO, SPACE]

    def __init__(self,
                 character_dict_path=None,
                 use_space_char=False,
                 **kwargs):
        super(UniRecLabelDecode, self).__init__(character_dict_path,
                                                use_space_char)
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            './configs/rec/unirec/unirec_100m')

    def __call__(self, preds, batch=None, *args, **kwargs):
        result_list = []
        pred_ids = preds
        res = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=False)
        for i in range(len(res)):
            res[i] = res[i].replace(' ', '').replace('Ġ', ' ').replace(
                'Ċ', '\n').replace('<|bos|>',
                                   '').replace('<|eos|>',
                                               '').replace('<|pad|>', '')
            result_list.append(
                (res[i],
                 0.0))  # Assuming confidence is not available, set to 0.0
        return result_list

    def add_special_char(self, dict_character):
        dict_character = self.list_token + dict_character
        return dict_character
