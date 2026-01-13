import torch
from transformers import PreTrainedTokenizerFast
from .ctc_postprocess import BaseRecLabelDecode


class CMERLabelDecode(BaseRecLabelDecode):
    """
    Decodes model output Token IDs into text.
    Refactored to match UniRecLabelDecode style and return format (text, score).
    """

    def __init__(
            self,
            character_dict_path=None,
            use_space_char=False,
            tokenizer_file='./configs/rec/cmer/cmer_tokenizer/tokenizer.json',
            **kwargs):
        """
        Args:
            character_dict_path: Path to character dict (inherited param).
            use_space_char: Whether to use space char (inherited param).
            tokenizer_file: Path to the tokenizer json file.
            **kwargs: Other configurations.
        """
        # 1. Call super constructor to match UniRec style
        super(CMERLabelDecode, self).__init__(character_dict_path,
                                              use_space_char)

        # 2. CMER specific logic
        self.remove_spaces = True

        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=tokenizer_file,
            padding_side='right',
            truncation_side='right',
            pad_token='<|pad|>',
            bos_token='<|bos|>',
            eos_token='<|eos|>',
            unk_token='<|unk|>',
        )

    def get_character_num(self):
        """
        Called by Trainer to determine classification layer output dimension (vocab_size).
        """
        if hasattr(self.tokenizer, 'vocab_size'):
            return self.tokenizer.vocab_size
        elif hasattr(self.tokenizer, '__len__'):
            return len(self.tokenizer)
        return 0

    def __call__(self, preds, batch=None, *args, **kwargs):
        """
        Args:
            preds: Tensor (Batch, Seq_Len) or Dict
            batch: Raw batch data
        Returns:
            list: List of tuples [(text, score), ...]
        """
        # Handle tuple/dict inputs from Trainer
        if isinstance(preds, dict):
            if 'cmer_pred' in preds:
                token_ids = preds['cmer_pred']
            elif 'maps' in preds:
                token_ids = preds['maps']
            else:
                token_ids = next(iter(preds.values()))
        else:
            token_ids = preds

        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.cpu()

        # Batch decode using tokenizer
        decoded_texts = self.tokenizer.batch_decode(token_ids,
                                                    skip_special_tokens=True)

        result_list = []
        for text in decoded_texts:
            if self.remove_spaces:
                text = text.replace(' ', '')
            text = text.strip()

            result_list.append((text, 0.0))

        return result_list
