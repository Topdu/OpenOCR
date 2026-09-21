"""
Script-Aware AR Label Encode for Multilingual Text Recognition
with Independent Expert Dictionaries.

Extends ARLabelEncode to:
1. Detect the dominant script group of each sample
2. Encode labels using the corresponding expert's dictionary
3. Output script_id for routing supervision

Expert groups (4 groups):
    0: Latin + Cyrillic
    1: CJK (Chinese, Japanese, Korean)
    2: Arabic + RTL
    3: Indic + SE-Asian (Hindi, Bangla, Thai, Tibetan)

Each expert has its own character dictionary, so the label indices are
specific to that expert's vocabulary. The decoder uses the same expert's
embedding and projection during training and inference.
"""

import unicodedata

import numpy as np

from openrec.preprocess.ctc_label_encode import BaseRecLabelEncode


# ============================================================================
# Unicode-based Script Detection
# ============================================================================

_SCRIPT_RANGES = {
    # Group 0: Latin + Cyrillic
    'latin_cyrillic': [
        (0x0000, 0x024F),   # Basic Latin, Latin Extended-A/B
        (0x1E00, 0x1EFF),   # Latin Extended Additional
        (0x0400, 0x04FF),   # Cyrillic
        (0x0500, 0x052F),   # Cyrillic Supplement
        (0x2DE0, 0x2DFF),   # Cyrillic Extended-A
        (0xA640, 0xA69F),   # Cyrillic Extended-B
    ],
    # Group 1: CJK
    'cjk': [
        (0x4E00, 0x9FFF),   # CJK Unified Ideographs
        (0x3400, 0x4DBF),   # CJK Unified Ideographs Extension A
        (0x20000, 0x2A6DF), # CJK Unified Ideographs Extension B
        (0x3000, 0x303F),   # CJK Symbols and Punctuation
        (0x3040, 0x309F),   # Hiragana
        (0x30A0, 0x30FF),   # Katakana
        (0xAC00, 0xD7AF),   # Hangul Syllables
        (0x1100, 0x11FF),   # Hangul Jamo
        (0x3130, 0x318F),   # Hangul Compatibility Jamo
    ],
    # Group 2: Arabic + RTL
    'arabic_rtl': [
        (0x0600, 0x06FF),   # Arabic
        (0x0750, 0x077F),   # Arabic Supplement
        (0x0870, 0x089F),   # Arabic Extended-B
        (0x08A0, 0x08FF),   # Arabic Extended-A
        (0xFB50, 0xFDFF),   # Arabic Presentation Forms-A
        (0xFE70, 0xFEFF),   # Arabic Presentation Forms-B
        (0x0590, 0x05FF),   # Hebrew
        (0x0700, 0x074F),   # Syriac
        (0x0780, 0x07BF),   # Thaana
        (0x07C0, 0x07FF),   # NKo
    ],
    # Group 3: Indic + SE-Asian
    'indic_se_asian': [
        (0x0900, 0x097F),   # Devanagari (Hindi)
        (0x0980, 0x09FF),   # Bengali/Bangla
        (0x0A00, 0x0A7F),   # Gurmukhi
        (0x0A80, 0x0AFF),   # Gujarati
        (0x0B00, 0x0B7F),   # Oriya
        (0x0B80, 0x0BFF),   # Tamil
        (0x0C00, 0x0C7F),   # Telugu
        (0x0C80, 0x0CFF),   # Kannada
        (0x0D00, 0x0D7F),   # Malayalam
        (0x0D80, 0x0DFF),   # Sinhala
        (0x0E00, 0x0E7F),   # Thai
        (0x0E80, 0x0EFF),   # Lao
        (0x0F00, 0x0FFF),   # Tibetan
        (0x1000, 0x109F),   # Myanmar
        (0x1780, 0x17FF),   # Khmer
    ],
}

_GROUP_NAMES = list(_SCRIPT_RANGES.keys())
_GROUP_ID = {name: idx for idx, name in enumerate(_GROUP_NAMES)}


def _build_lookup_table():
    ranges_with_ids = []
    for group_name, ranges in _SCRIPT_RANGES.items():
        gid = _GROUP_ID[group_name]
        for start, end in ranges:
            ranges_with_ids.append((start, end, gid))
    ranges_with_ids.sort(key=lambda x: x[0])
    return ranges_with_ids


_LOOKUP = _build_lookup_table()


def get_script_id(char):
    cp = ord(char)
    lo, hi = 0, len(_LOOKUP) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        start, end, gid = _LOOKUP[mid]
        if cp < start:
            hi = mid - 1
        elif cp > end:
            lo = mid + 1
        else:
            return gid
    return -1


def detect_script(text, num_groups=4):
    if not text:
        return 0
    counts = [0] * num_groups
    for char in text:
        gid = get_script_id(char)
        if 0 <= gid < num_groups:
            counts[gid] += 1
    if sum(counts) == 0:
        for char in text:
            try:
                name = unicodedata.name(char, '').upper()
                if 'CJK' in name or 'HANGUL' in name or 'HIRAGANA' in name or 'KATAKANA' in name:
                    return 1
                elif 'ARABIC' in name or 'HEBREW' in name:
                    return 2
                elif 'DEVANAGARI' in name or 'BENGALI' in name or 'THAI' in name or 'TIBETAN' in name:
                    return 3
            except ValueError:
                continue
        return 0
    return int(np.argmax(counts))


# ============================================================================
# Helper: load a character dictionary
# ============================================================================
def _load_dict(dict_path, use_space_char=False):
    """Load character dict and return: (char_list, char2idx, idx2char)

    The returned dict includes special tokens:
        index 0: EOS ('</s>')
        index 1..N: characters
        index N+1: BOS ('<s>')
        index N+2: PAD ('<pad>')
    """
    chars = []
    with open(dict_path, 'rb') as f:
        for line in f.readlines():
            line = line.decode('utf-8').strip('\n').strip('\r\n')
            if line:
                chars.append(line)
    if use_space_char:
        chars.append(' ')

    EOS = '</s>'
    BOS = '<s>'
    PAD = '<pad>'
    # Same convention as BaseRecLabelEncode.add_special_char for AR:
    # [EOS] + chars + [BOS, PAD]
    full_chars = [EOS] + chars + [BOS, PAD]
    char2idx = {c: i for i, c in enumerate(full_chars)}
    return full_chars, char2idx


# ============================================================================
# Script-Aware AR Label Encode with Independent Expert Dictionaries
# ============================================================================

class ScriptAwareARLabelEncode(BaseRecLabelEncode):
    """AR Label Encode with per-expert independent dictionaries.

    Each sample is:
    1. Script-detected to determine which expert it belongs to
    2. Encoded using that expert's specific dictionary
    3. Annotated with script_id for routing supervision

    If expert_dict_paths is not provided, falls back to the shared dictionary
    (backward compatible with the original behavior).

    Args:
        max_text_length (int): Maximum text length.
        character_dict_path (str): Path to shared/fallback character dictionary.
        use_space_char (bool): Whether to include space.
        num_script_groups (int): Number of script groups (default 4).
        expert_dict_paths (list[str]|None): Paths to per-expert dictionaries.
            If provided, must have length == num_script_groups.
    """

    BOS = '<s>'
    EOS = '</s>'
    PAD = '<pad>'

    def __init__(
        self,
        max_text_length,
        character_dict_path=None,
        use_space_char=False,
        num_script_groups=4,
        expert_dict_paths=None,
        **kwargs,
    ):
        # Initialize with shared dict (for fallback / compatibility)
        super().__init__(
            max_text_length, character_dict_path, use_space_char)
        self.num_script_groups = num_script_groups

        # Load per-expert dictionaries if provided
        self.use_independent_vocab = expert_dict_paths is not None
        if self.use_independent_vocab:
            assert len(expert_dict_paths) == num_script_groups, \
                f"expert_dict_paths ({len(expert_dict_paths)}) != num_script_groups ({num_script_groups})"

            self.expert_dicts = []  # list of (full_chars, char2idx)
            for path in expert_dict_paths:
                full_chars, char2idx = _load_dict(path, use_space_char)
                self.expert_dicts.append((full_chars, char2idx))

    def _encode_with_expert_dict(self, text, expert_id):
        """Encode text using a specific expert's dictionary.

        Returns list of character indices, or None if text is empty/too long.
        """
        full_chars, char2idx = self.expert_dicts[expert_id]
        text_list = []
        for char in text:
            if char in char2idx:
                text_list.append(char2idx[char])
            # Characters not in this expert's dict are silently dropped
        if len(text_list) == 0 or len(text_list) > self.max_text_len:
            return None
        return text_list

    def __call__(self, data):
        text = data['label']

        # Detect script before encoding
        script_id = detect_script(text, self.num_script_groups)

        if self.use_independent_vocab:
            # Encode with expert-specific dictionary
            text_encoded = self._encode_with_expert_dict(text, script_id)
            if text_encoded is None:
                return None

            full_chars, char2idx = self.expert_dicts[script_id]
            bos_idx = char2idx[self.BOS]
            eos_idx = char2idx[self.EOS]
            pad_idx = char2idx[self.PAD]

            data['length'] = np.array(len(text_encoded))
            text_encoded = [bos_idx] + text_encoded + [eos_idx]
            text_encoded = text_encoded + [pad_idx] * (
                self.max_text_len + 2 - len(text_encoded)
            )
            data['label'] = np.array(text_encoded)
        else:
            # Fallback to shared dictionary
            text_encoded = self.encode(text)
            if text_encoded is None:
                return None

            data['length'] = np.array(len(text_encoded))
            text_encoded = (
                [self.dict[self.BOS]] + text_encoded + [self.dict[self.EOS]]
            )
            text_encoded = text_encoded + [self.dict[self.PAD]] * (
                self.max_text_len + 2 - len(text_encoded)
            )
            data['label'] = np.array(text_encoded)

        data['script_id'] = np.array(script_id, dtype=np.int64)
        return data

    def add_special_char(self, dict_character):
        dict_character = [self.EOS] + dict_character + [self.BOS, self.PAD]
        return dict_character
