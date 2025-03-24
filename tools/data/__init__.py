import os
import sys
import copy
import importlib

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../..')))

from torch.utils.data import DataLoader, DistributedSampler

# 定义支持的 Dataset 类及其对应的模块路径
DATASET_MODULES = {
    'SimpleDataSet': 'tools.data.simple_dataset',
    'LMDBDataSet': 'tools.data.lmdb_dataset',
    'TextLMDBDataSet': 'tools.data.text_lmdb_dataset',
    'MultiScaleDataSet': 'tools.data.simple_dataset',
    'STRLMDBDataSet': 'tools.data.strlmdb_dataset',
    'LMDBDataSetTest': 'tools.data.lmdb_dataset_test',
    'RatioDataSet': 'tools.data.ratio_dataset',
    'RatioDataSetTest': 'tools.data.ratio_dataset_test',
    'RatioDataSetTVResize': 'tools.data.ratio_dataset_tvresize',
    'RatioDataSetTVResizeTest': 'tools.data.ratio_dataset_tvresize_test'
}

# 定义支持的 Sampler 类及其对应的模块路径
SAMPLER_MODULES = {
    'MultiScaleSampler': 'tools.data.multi_scale_sampler',
    'RatioSampler': 'tools.data.ratio_sampler'
}

__all__ = [
    'build_dataloader',
]


def build_dataloader(config, mode, logger, seed=None, epoch=3, task='rec'):
    config = copy.deepcopy(config)
    mode = mode.capitalize()  # 确保 mode 是首字母大写形式（Train/Eval/Test）

    # 获取 dataset 配置
    dataset_config = config[mode]['dataset']
    module_name = dataset_config['name']

    # 动态导入 dataset 类
    if module_name not in DATASET_MODULES:
        raise ValueError(
            f'Unsupported dataset: {module_name}. Supported datasets: {list(DATASET_MODULES.keys())}'
        )

    dataset_module = importlib.import_module(DATASET_MODULES[module_name])
    dataset_class = getattr(dataset_module, module_name)
    dataset = dataset_class(config, mode, logger, seed, epoch=epoch, task=task)

    # DataLoader 配置
    loader_config = config[mode]['loader']
    batch_size = loader_config['batch_size_per_card']
    drop_last = loader_config['drop_last']
    shuffle = loader_config['shuffle']
    num_workers = loader_config['num_workers']
    pin_memory = loader_config.get('pin_memory', False)

    sampler = None
    batch_sampler = None
    if 'sampler' in config[mode]:
        sampler_config = config[mode]['sampler']
        sampler_name = sampler_config.pop('name')

        if sampler_name not in SAMPLER_MODULES:
            raise ValueError(
                f'Unsupported sampler: {sampler_name}. Supported samplers: {list(SAMPLER_MODULES.keys())}'
            )

        sampler_module = importlib.import_module(SAMPLER_MODULES[sampler_name])
        sampler_class = getattr(sampler_module, sampler_name)
        batch_sampler = sampler_class(dataset, **sampler_config)
    elif config['Global']['distributed'] and mode == 'Train':
        sampler = DistributedSampler(dataset=dataset, shuffle=shuffle)

    if 'collate_fn' in loader_config:
        from . import collate_fn
        collate_fn = getattr(collate_fn, loader_config['collate_fn'])()
    else:
        collate_fn = None

    if batch_sampler is None:
        data_loader = DataLoader(
            dataset=dataset,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
            batch_size=batch_size,
            drop_last=drop_last,
        )
    else:
        data_loader = DataLoader(
            dataset=dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
        )

    # 检查数据加载器是否为空
    if len(data_loader) == 0:
        logger.error(
            f'No Images in {mode.lower()} dataloader. Please check:\n'
            '\t1. The images num in the train label_file_list should be >= batch size.\n'
            '\t2. The annotation file and path in the configuration are correct.\n'
            '\t3. The BatchSize is not larger than the number of images.')
        sys.exit()

    return data_loader
