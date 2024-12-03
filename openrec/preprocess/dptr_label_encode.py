import re
from abc import ABC, abstractmethod
from itertools import groupby
from typing import List, Optional, Tuple
import numpy as np
import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
import unicodedata
from ..modeling.decoders.dptr_parseq_clip_b_decoder import tokenize

class CharsetAdapter:
    """Transforms labels according to the target charset."""

    def __init__(self, target_charset) -> None:
        super().__init__()
        self.lowercase_only = target_charset == target_charset.lower()
        self.uppercase_only = target_charset == target_charset.upper()
        self.unsupported = re.compile(f'[^{re.escape(target_charset)}]')

    def __call__(self, label):
        if self.lowercase_only:
            label = label.lower()
        elif self.uppercase_only:
            label = label.upper()
        # Remove unsupported characters
        label = self.unsupported.sub('', label)
        return label


class BaseTokenizer(ABC):
# eos=0, a=1, bos=37, pad=38
    def __init__(self, charset: str, specials_first: tuple = (), specials_last: tuple = ()) -> None:
        self._itos = specials_first + tuple(charset) + specials_last
        self._stoi = {s: i for i, s in enumerate(self._itos)}
        # print("stoi:", self._stoi)

    def __len__(self):
        return len(self._itos)

    def _tok2ids(self, tokens: str) -> List[int]:
        # print("tokens", tokens)
        return [self._stoi[s] for s in tokens]

    def _ids2tok(self, token_ids: List[int], join: bool = True) -> str:
        tokens = [self._itos[i] for i in token_ids]
        return ''.join(tokens) if join else tokens

    @abstractmethod
    def encode(self, labels: List[str], device: Optional[torch.device] = None) -> Tensor:
        """Encode a batch of labels to a representation suitable for the model.

        Args:
            labels: List of labels. Each can be of arbitrary length.
            device: Create tensor on this device.

        Returns:
            Batched tensor representation padded to the max label length. Shape: N, L
        """
        raise NotImplementedError

    @abstractmethod
    def _filter(self, probs: Tensor, ids: Tensor) -> Tuple[Tensor, List[int]]:
        """Internal method which performs the necessary filtering prior to decoding."""
        raise NotImplementedError

    def decode(self, token_dists: Tensor, raw: bool = False) -> Tuple[List[str], List[Tensor]]:
        """Decode a batch of token distributions.

        Args:
            token_dists: softmax probabilities over the token distribution. Shape: N, L, C
            raw: return unprocessed labels (will return list of list of strings)

        Returns:
            list of string labels (arbitrary length) and
            their corresponding sequence probabilities as a list of Tensors
        """
        batch_tokens = []
        batch_probs = []
        for dist in token_dists:
            probs, ids = dist.max(-1)  # greedy selection
            if not raw:
                probs, ids = self._filter(probs, ids)
            tokens = self._ids2tok(ids, not raw)
            batch_tokens.append(tokens)
            batch_probs.append(probs)
        return batch_tokens, batch_probs


class Tokenizer(BaseTokenizer):
    BOS = '[B]'
    EOS = '[E]'
    PAD = '[P]'

    def __init__(self, charset: str) -> None:
        specials_first = (self.EOS,)
        specials_last = (self.BOS, self.PAD)
        super().__init__(charset, specials_first, specials_last)
        self.eos_id, self.bos_id, self.pad_id = [self._stoi[s] for s in specials_first + specials_last]

    def encode(self, labels: List[str], device: Optional[torch.device] = None) -> Tensor:
        batch = [self.bos_id] + self._tok2ids(labels) + [self.eos_id]
        return batch
        # return pad_sequence(batch, batch_first=True, padding_value=self.pad_id)

    def _filter(self, probs: Tensor, ids: Tensor) -> Tuple[Tensor, List[int]]:
        ids = ids.tolist()
        try:
            eos_idx = ids.index(self.eos_id)
        except ValueError:
            eos_idx = len(ids)  # Nothing to truncate.
        # Truncate after EOS
        ids = ids[:eos_idx]
        probs = probs[:eos_idx + 1]  # but include prob. for EOS (if it exists)
        return probs, ids

class DPTRLabelEncode(Tokenizer):
    """Convert between text-label and text-index."""
    def __init__(self, max_text_length=25, character_dict_path=None, **kwargs):
        self.max_length = max_text_length
        charset = get_alpha(character_dict_path)
        charset = ''.join(charset)
        # print(charset)
        super(DPTRLabelEncode, self).__init__(charset)

    def __call__(self, data, normalize_unicode=True):
        text = data['label']

        if normalize_unicode:
            text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode()
        text = ''.join(text.split())
        if len(text) == 0 or len(text) > self.max_length:
            return None

        text_ids = self.encode(text)
        clip_ids = tokenize(f"a photo of a '{text}'")
        text_ids = text_ids + [self.pad_id] * (self.max_length + 2 - len(text_ids))
        # print(text, len(text_ids), len(clip_ids[0]))
        data['clip_label'] = np.array(clip_ids[0])
        data['label'] = np.array(text_ids)
        return data

    def add_special_char(self, dict_character):
        dict_character = [self.EOS] + dict_character + [self.BOS, self.PAD]
        return dict_character

def get_alpha(alpha_path):
    character_str = []
    with open(alpha_path, 'rb') as fin:
        lines = fin.readlines()
        for line in lines:
            line = line.decode('utf-8').strip('\n').strip('\r\n')
            character_str.append(line)
    dict_character = list(character_str)
    if 'arabic' in alpha_path:
        reverse = True
    return dict_character