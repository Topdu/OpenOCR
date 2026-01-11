import json
import os
import random
import traceback
import importlib
import numpy as np
from PIL import Image, ImageFile
from torch.utils.data import Dataset

# 防止 PIL 加载截断的图片报错
ImageFile.LOAD_TRUNCATED_IMAGES = True


class RawImageDataSet(Dataset):
    """
    RawImageDataSet:
    只负责读取原始图片和标签。
    预处理委托给 loader_config 中定义的 collate_fn (Processor)。
    """

    def __init__(self, config, mode, logger, seed=None, epoch=0, task='rec'):
        super(RawImageDataSet, self).__init__()
        self.logger = logger
        self.mode = mode.lower()

        dataset_config = config[mode]['dataset']
        loader_config = config[mode]['loader']

        self.delimiter = dataset_config.get('delimiter', '\t')
        label_file_list = dataset_config.pop('label_file_list')
        data_source_num = len(label_file_list)
        ratio_list = dataset_config.get('ratio_list', 1.0)

        if isinstance(ratio_list, (float, int)):
            ratio_list = [float(ratio_list)] * int(data_source_num)

        assert len(ratio_list) == data_source_num, \
            'The length of ratio_list should be the same as the file_list.'

        self.data_dir = dataset_config['data_dir']
        self.do_shuffle = loader_config['shuffle']
        self.seed = seed

        logger.info(f'Initialize indexs of datasets: {label_file_list}')
        self.data_lines = self.get_image_info_list(label_file_list, ratio_list)
        self.data_idx_order_list = list(range(len(self.data_lines)))

        if self.mode == 'train' and self.do_shuffle:
            self.shuffle_data_random()

        self.ext_op_transform_idx = dataset_config.get('ext_op_transform_idx',
                                                       2)
        self.need_reset = True in [x < 1 for x in ratio_list]

        # === 初始化 Collate Fn (Processor) 并进行适配 ===
        self.collate_fn = None
        if 'collate_fn' in loader_config:
            collate_fn_name = loader_config['collate_fn']

            if isinstance(collate_fn_name,
                          str) and collate_fn_name.endswith('Processor'):
                self.logger.info(
                    f'Detected Processor as collate_fn: {collate_fn_name}')
                module_path = loader_config.get('collate_fn_source')

                if module_path and module_path.startswith('.'):
                    module_path = module_path.lstrip('.')

                try:
                    proc_module = importlib.import_module(module_path)
                    ProcessorClass = getattr(proc_module, collate_fn_name)
                    proc_args = loader_config.get('processor_args', {})

                    self.logger.info(
                        f'Initializing {collate_fn_name} with args: {proc_args}'
                    )

                    # 1. 实例化 Processor
                    self.processor = ProcessorClass(**proc_args)

                    # 2. 定义 Wrapper：将 Processor 的 Dict 输出转换为 List 输出
                    # DataLoader 会传入一个 batch 的 list (来自 __getitem__)
                    def collate_wrapper(batch):
                        # 调用 Processor，得到 {'pixel_values': ..., 'labels': ..., 'length': ...}
                        out_dict = self.processor(batch)

                        # 按模型 forward 需要的顺序，解包成列表 [images, labels, lengths]
                        return [
                            out_dict['image'], out_dict['label'],
                            out_dict['length']
                        ]

                    # 3. 将 Wrapper 赋值给 self.collate_fn
                    self.collate_fn = collate_wrapper

                except Exception as e:
                    self.logger.error(
                        f'Failed to initialize processor {collate_fn_name}: {e}'
                    )
                    raise e
            else:
                # 兼容旧逻辑
                pass

    def get_image_info_list(self, file_list, ratio_list):
        if isinstance(file_list, str):
            file_list = [file_list]
        data_lines = []
        for idx, file in enumerate(file_list):
            with open(file, 'rb') as f:
                lines = f.readlines()
                if self.mode == 'train' or ratio_list[idx] < 1.0:
                    random.seed(self.seed)
                    lines = random.sample(lines,
                                          round(len(lines) * ratio_list[idx]))
                data_lines.extend(lines)
        return data_lines

    def shuffle_data_random(self):
        random.seed(self.seed)
        random.shuffle(self.data_lines)
        return

    def _try_parse_filename_list(self, file_name):
        if len(file_name) > 0 and file_name[0] == '[':
            try:
                info = json.loads(file_name)
                file_name = random.choice(info)
            except:
                pass
        return file_name

    def __getitem__(self, idx):
        file_idx = self.data_idx_order_list[idx]
        data_line = self.data_lines[file_idx]
        try:
            data_line = data_line.decode('utf-8')
            substr = data_line.strip('\n').split(self.delimiter)
            file_name = substr[0]
            file_name = self._try_parse_filename_list(file_name)
            label = substr[1]
            img_path = os.path.join(self.data_dir, file_name)

            if not os.path.exists(img_path):
                raise Exception(f'{img_path} does not exist!')

            image = Image.open(img_path).convert('RGB')

            return {'image': image, 'label': label, 'img_path': img_path}

        except Exception:
            self.logger.error(
                'When parsing line {}, error happened with msg: {}'.format(
                    data_line, traceback.format_exc()))
            rnd_idx = np.random.randint(self.__len__(
            )) if self.mode == 'train' else (idx + 1) % self.__len__()
            return self.__getitem__(rnd_idx)

    def __len__(self):
        return len(self.data_idx_order_list)
