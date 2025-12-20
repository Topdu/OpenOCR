import re
from .ctc_postprocess import BaseRecLabelDecode

rules = [
    (r'-<\|sn\|>', ''),
    (r' <\|sn\|>', ' '),
    (r'<\|sn\|>', ' '),
    (r'<\|unk\|>', ''),
    (r'<s>', ''),
    (r'</s>', ''),
    (r'\uffff', ''),
    (r'_{4,}', '___'),
    (r'\.{4,}', '...'),
]


def clean_special_tokens(text):
    text = text.replace(' ', '').replace('Ġ', ' ').replace('Ċ', '\n').replace(
        '<|bos|>', '').replace('<|eos|>', '').replace('<|pad|>', '')
    for rule in rules:
        text = re.sub(rule[0], rule[1], text)
    text = text.replace('<tdcolspan=', '<td colspan=')
    text = text.replace('<tdrowspan=', '<td rowspan=')
    text = text.replace('"colspan=', '" colspan=')
    return text


class UniRecLabelDecode(BaseRecLabelDecode):
    """Convert between text-label and text-index."""
    SPACE = '[s]'
    GO = '[GO]'
    list_token = [GO, SPACE]

    def __init__(self,
                 character_dict_path=None,
                 use_space_char=False,
                 tokenizer_path='./configs/rec/unirec/unirec-0.1b',
                 **kwargs):
        super(UniRecLabelDecode, self).__init__(character_dict_path,
                                                use_space_char)
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    def __call__(self, preds, batch=None, *args, **kwargs):
        result_list = []
        pred_ids = preds
        res = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=False)
        for i in range(len(res)):
            res[i] = clean_special_tokens(res[i])
            result_list.append(
                (res[i],
                 0.0))  # Assuming confidence is not available, set to 0.0
        return result_list

    def add_special_char(self, dict_character):
        dict_character = self.list_token + dict_character
        return dict_character
