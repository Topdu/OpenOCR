import io
import re
import unicodedata

import lmdb
from PIL import Image
from torch.utils.data import Dataset

from openrec.preprocess import create_operators, transform


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


class LMDBDataSetTest(Dataset):
    """Dataset interface to an LMDB database.

    It supports both labelled and unlabelled datasets. For unlabelled datasets,
    the image index itself is returned as the label. Unicode characters are
    normalized by default. Case-sensitivity is inferred from the charset.
    Labels are transformed according to the charset.
    """

    def __init__(self,
                 config,
                 mode,
                 logger,
                 seed=None,
                 epoch=1,
                 gpu_i=0,
                 max_label_len: int = 25,
                 min_image_dim: int = 0,
                 remove_whitespace: bool = True,
                 normalize_unicode: bool = True,
                 unlabelled: bool = False,
                 transform=None,
                 task='rec'):
        dataset_config = config[mode]['dataset']
        global_config = config['Global']
        max_label_len = global_config['max_text_length']
        self.root = dataset_config['data_dir']
        self._env = None
        self.unlabelled = unlabelled
        self.transform = transform
        self.labels = []
        self.filtered_index_list = []
        self.min_image_dim = min_image_dim
        self.filter_label = dataset_config.get('filter_label',
                                               True)  #'data_dir']filter_label
        character_dict_path = global_config.get('character_dict_path', None)
        use_space_char = global_config.get('use_space_char', False)
        if character_dict_path is None:
            char_test = '0123456789abcdefghijklmnopqrstuvwxyz'
        else:
            char_test = ''
            with open(character_dict_path, 'rb') as fin:
                lines = fin.readlines()
                for line in lines:
                    line = line.decode('utf-8').strip('\n').strip('\r\n')
                    char_test += line
            if use_space_char:
                char_test += ' '
        self.ops = create_operators(dataset_config['transforms'],
                                    global_config)
        self.num_samples = self._preprocess_labels(char_test,
                                                   remove_whitespace,
                                                   normalize_unicode,
                                                   max_label_len,
                                                   min_image_dim)

    def __del__(self):
        if self._env is not None:
            self._env.close()
            self._env = None

    def _create_env(self):
        return lmdb.open(self.root,
                         max_readers=1,
                         readonly=True,
                         create=False,
                         readahead=False,
                         meminit=False,
                         lock=False)

    @property
    def env(self):
        if self._env is None:
            self._env = self._create_env()
        return self._env

    def _preprocess_labels(self, charset, remove_whitespace, normalize_unicode,
                           max_label_len, min_image_dim):
        charset_adapter = CharsetAdapter(charset)
        with self._create_env() as env, env.begin() as txn:
            num_samples = int(txn.get('num-samples'.encode()))
            if self.unlabelled:
                return num_samples
            for index in range(num_samples):
                index += 1  # lmdb starts with 1
                label_key = f'label-{index:09d}'.encode()
                label = txn.get(label_key).decode()
                # Normally, whitespace is removed from the labels.
                if remove_whitespace:
                    label = ''.join(label.split())
                # Normalize unicode composites (if any) and convert to compatible ASCII characters
                if self.filter_label:
                    # if normalize_unicode:
                    label = unicodedata.normalize('NFKD', label).encode(
                        'ascii', 'ignore').decode()
                # Filter by length before removing unsupported characters. The original label might be too long.
                if len(label) > max_label_len:
                    continue

                if self.filter_label:
                    label = charset_adapter(label)
                # We filter out samples which don't contain any supported characters
                if not label:
                    continue
                # Filter images that are too small.
                if min_image_dim > 0:
                    img_key = f'image-{index:09d}'.encode()
                    img = txn.get(img_key)
                    data = {'image': img, 'label': label}
                    outs = transform(data, self.ops)
                    if outs is None:
                        continue
                    buf = io.BytesIO(img)
                    w, h = Image.open(buf).size
                    if w < self.min_image_dim or h < self.min_image_dim:
                        continue
                self.labels.append(label)
                self.filtered_index_list.append(index)
        return len(self.labels)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        if self.unlabelled:
            label = index
        else:
            label = self.labels[index]
            index = self.filtered_index_list[index]

        img_key = f'image-{index:09d}'.encode()
        with self.env.begin() as txn:
            img = txn.get(img_key)
        data = {'image': img, 'label': label}
        outs = transform(data, self.ops)

        return outs
