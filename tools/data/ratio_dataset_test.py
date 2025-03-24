import io
import math
import random
import re
import unicodedata

import cv2
import lmdb
import numpy as np
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


class RatioDataSetTest(Dataset):

    def __init__(self, config, mode, logger, seed=None, epoch=1, task='rec'):
        super(RatioDataSetTest, self).__init__()
        self.ds_width = config[mode]['dataset'].get('ds_width', True)
        global_config = config['Global']
        dataset_config = config[mode]['dataset']
        loader_config = config[mode]['loader']
        max_ratio = loader_config.get('max_ratio', 10)
        min_ratio = loader_config.get('min_ratio', 1)
        data_dir_list = dataset_config['data_dir_list']
        self.do_shuffle = loader_config['shuffle']
        self.seed = epoch
        self.max_text_length = global_config['max_text_length']
        data_source_num = len(data_dir_list)
        ratio_list = dataset_config.get('ratio_list', 1.0)
        if isinstance(ratio_list, (float, int)):
            ratio_list = [float(ratio_list)] * int(data_source_num)
        assert len(
            ratio_list
        ) == data_source_num, 'The length of ratio_list should be the same as the file_list.'
        self.lmdb_sets = self.load_hierarchical_lmdb_dataset(
            data_dir_list, ratio_list)
        for data_dir in data_dir_list:
            logger.info('Initialize indexs of datasets:%s' % data_dir)
        self.logger = logger
        data_idx_order_list = self.dataset_traversal()
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
        wh_ratio, data_idx_order_list = self.get_wh_ratio(
            data_idx_order_list, char_test)
        self.data_idx_order_list = np.array(data_idx_order_list)
        wh_ratio = np.around(np.array(wh_ratio))
        self.wh_ratio = np.clip(wh_ratio, a_min=min_ratio, a_max=max_ratio)
        for i in range(max_ratio + 1):
            logger.info((1 * (self.wh_ratio == i)).sum())
        self.wh_ratio_sort = np.argsort(self.wh_ratio)
        self.ops = create_operators(dataset_config['transforms'],
                                    global_config)

        self.need_reset = True in [x < 1 for x in ratio_list]
        self.error = 0
        self.base_shape = dataset_config.get(
            'base_shape', [[64, 64], [96, 48], [112, 40], [128, 32]])
        self.base_h = 32

    def get_wh_ratio(self, data_idx_order_list, char_test):
        wh_ratio = []
        wh_ratio_len = [[0 for _ in range(26)] for _ in range(11)]
        data_idx_order_list_filter = []
        charset_adapter = CharsetAdapter(char_test)

        for idx in range(data_idx_order_list.shape[0]):
            lmdb_idx, file_idx = data_idx_order_list[idx]
            lmdb_idx = int(lmdb_idx)
            file_idx = int(file_idx)
            wh_key = 'wh-%09d'.encode() % file_idx
            wh = self.lmdb_sets[lmdb_idx]['txn'].get(wh_key)
            if wh is None:
                img_key = f'image-{file_idx:09d}'.encode()
                img = self.lmdb_sets[lmdb_idx]['txn'].get(img_key)
                buf = io.BytesIO(img)
                w, h = Image.open(buf).size
            else:
                wh = wh.decode('utf-8')
                w, h = wh.split('_')

            label_key = 'label-%09d'.encode() % file_idx
            label = self.lmdb_sets[lmdb_idx]['txn'].get(label_key)
            if label is not None:
                # return None
                label = label.decode('utf-8')
                # if remove_whitespace:
                label = ''.join(label.split())
                # Normalize unicode composites (if any) and convert to compatible ASCII characters
                # if normalize_unicode:
                label = unicodedata.normalize('NFKD',
                                              label).encode('ascii',
                                                            'ignore').decode()
                # Filter by length before removing unsupported characters. The original label might be too long.
                if len(label) > self.max_text_length:
                    continue
                label = charset_adapter(label)
                if not label:
                    continue

                wh_ratio.append(float(w) / float(h))
                wh_ratio_len[int(float(w) /
                                 float(h)) if int(float(w) /
                                                  float(h)) <= 10 else
                             10][len(label) if len(label) <= 25 else 25] += 1
                data_idx_order_list_filter.append([lmdb_idx, file_idx])
        self.logger.info(wh_ratio_len)
        return wh_ratio, data_idx_order_list_filter

    def load_hierarchical_lmdb_dataset(self, data_dir_list, ratio_list):
        lmdb_sets = {}
        dataset_idx = 0
        for dirpath, ratio in zip(data_dir_list, ratio_list):
            env = lmdb.open(dirpath,
                            max_readers=32,
                            readonly=True,
                            lock=False,
                            readahead=False,
                            meminit=False)
            txn = env.begin(write=False)
            num_samples = int(txn.get('num-samples'.encode()))
            lmdb_sets[dataset_idx] = {
                'dirpath': dirpath,
                'env': env,
                'txn': txn,
                'num_samples': num_samples,
                'ratio_num_samples': int(ratio * num_samples),
            }
            dataset_idx += 1
        return lmdb_sets

    def dataset_traversal(self):
        lmdb_num = len(self.lmdb_sets)
        total_sample_num = 0
        for lno in range(lmdb_num):
            total_sample_num += self.lmdb_sets[lno]['ratio_num_samples']
        data_idx_order_list = np.zeros((total_sample_num, 2))
        beg_idx = 0
        for lno in range(lmdb_num):
            tmp_sample_num = self.lmdb_sets[lno]['ratio_num_samples']
            end_idx = beg_idx + tmp_sample_num
            data_idx_order_list[beg_idx:end_idx, 0] = lno
            data_idx_order_list[beg_idx:end_idx, 1] = list(
                random.sample(range(1, self.lmdb_sets[lno]['num_samples'] + 1),
                              self.lmdb_sets[lno]['ratio_num_samples']))
            beg_idx = beg_idx + tmp_sample_num
        return data_idx_order_list

    def get_img_data(self, value):
        """get_img_data."""
        if not value:
            return None
        imgdata = np.frombuffer(value, dtype='uint8')
        if imgdata is None:
            return None
        imgori = cv2.imdecode(imgdata, 1)
        if imgori is None:
            return None
        return imgori

    def resize_norm_img(self, data, gen_ratio, padding=True):
        img = data['image']
        h = img.shape[0]
        w = img.shape[1]

        imgW, imgH = self.base_shape[gen_ratio - 1] if gen_ratio <= 4 else [
            self.base_h * gen_ratio, self.base_h
        ]
        use_ratio = imgW // imgH
        if use_ratio >= (w // h) + 2:
            self.error += 1
            return None
        if not padding:
            resized_image = cv2.resize(img, (imgW, imgH),
                                       interpolation=cv2.INTER_LINEAR)
            resized_w = imgW
        else:
            ratio = w / float(h)
            if math.ceil(imgH * ratio) > imgW:
                resized_w = imgW
            else:
                resized_w = int(
                    math.ceil(imgH * ratio * (random.random() + 0.5)))
                resized_w = min(imgW, resized_w)

            resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resized_image.astype('float32')
        resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        padding_im = np.zeros((3, imgH, imgW), dtype=np.float32)
        padding_im[:, :, :resized_w] = resized_image
        valid_ratio = min(1.0, float(resized_w / imgW))
        data['image'] = padding_im
        data['valid_ratio'] = valid_ratio
        data['gen_ratio'] = imgW // imgH
        data['real_ratio'] = max(1, round(w / h))
        return data

    def get_lmdb_sample_info(self, txn, index):
        label_key = 'label-%09d'.encode() % index
        label = txn.get(label_key)
        if label is None:
            return None
        label = label.decode('utf-8')
        img_key = 'image-%09d'.encode() % index
        imgbuf = txn.get(img_key)
        return imgbuf, label

    def __getitem__(self, properties):
        img_width = properties[0]
        img_height = properties[1]
        idx = properties[2]
        ratio = properties[3]
        lmdb_idx, file_idx = self.data_idx_order_list[idx]
        lmdb_idx = int(lmdb_idx)
        file_idx = int(file_idx)
        sample_info = self.get_lmdb_sample_info(
            self.lmdb_sets[lmdb_idx]['txn'], file_idx)
        if sample_info is None:
            ratio_ids = np.where(self.wh_ratio == ratio)[0].tolist()
            ids = random.sample(ratio_ids, 1)
            return self.__getitem__([img_width, img_height, ids[0], ratio])
        img, label = sample_info
        data = {'image': img, 'label': label}
        outs = transform(data, self.ops[:-1])
        if outs is not None:
            outs = self.resize_norm_img(outs, ratio, padding=False)
            if outs is None:
                ratio_ids = np.where(self.wh_ratio == ratio)[0].tolist()
                ids = random.sample(ratio_ids, 1)
                return self.__getitem__([img_width, img_height, ids[0], ratio])

            outs = transform(outs, self.ops[-1:])
        if outs is None:
            ratio_ids = np.where(self.wh_ratio == ratio)[0].tolist()
            ids = random.sample(ratio_ids, 1)
            return self.__getitem__([img_width, img_height, ids[0], ratio])
        return outs

    def __len__(self):
        return self.data_idx_order_list.shape[0]
