import os
from transformers import PreTrainedTokenizerFast

class CMERTokenizerBuilder:
    """
    Builder for the Tokenizer required by CMER
    """
    def __init__(self, tokenizer_path, **kwargs):
        self.tokenizer_path = tokenizer_path
        self.kwargs = kwargs

    def build(self):
        if not os.path.exists(self.tokenizer_path):
            raise FileNotFoundError(f"Tokenizer file not found at: {self.tokenizer_path}")
            
        # Extract special token parameters, pass the rest to the tokenizer
        special_tokens = {
            'pad_token': self.kwargs.pop('pad_token', '<|pad|>'),
            'bos_token': self.kwargs.pop('bos_token', '<|bos|>'),
            'eos_token': self.kwargs.pop('eos_token', '<|eos|>'),
            'unk_token': self.kwargs.pop('unk_token', '<|unk|>'),
        }

        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=self.tokenizer_path,
            **self.kwargs, # Pass in padding_side, truncation_side, etc.
            **special_tokens
        )
        return tokenizer