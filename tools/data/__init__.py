import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../..')))

import copy

from torch.utils.data import DataLoader, DistributedSampler

from tools.data.lmdb_dataset import LMDBDataSet
from tools.data.text_lmdb_dataset import TextLMDBDataSet
from tools.data.lmdb_dataset_test import LMDBDataSetTest
from tools.data.multi_scale_sampler import MultiScaleSampler
from tools.data.ratio_dataset import RatioDataSet
from tools.data.ratio_dataset_test import RatioDataSetTest
from tools.data.ratio_dataset_tvresize_test import RatioDataSetTVResizeTest
from tools.data.ratio_dataset_tvresize import RatioDataSetTVResize
from tools.data.ratio_sampler import RatioSampler
from tools.data.simple_dataset import MultiScaleDataSet, SimpleDataSet
from tools.data.strlmdb_dataset import STRLMDBDataSet

__all__ = [
    'build_dataloader',
    'transform',
    'create_operators',
]


def build_dataloader(config, mode, logger, seed=None, epoch=3):
    config = copy.deepcopy(config)

    support_dict = [
        'SimpleDataSet', 'LMDBDataSet', 'TextLMDBDataSet', 'MultiScaleDataSet', 'STRLMDBDataSet',
        'LMDBDataSetTest', 'RatioDataSet', 'RatioDataSetTest',
        'RatioDataSetTVResize', 'RatioDataSetTVResizeTest'
    ]
    module_name = config[mode]['dataset']['name']
    assert module_name in support_dict, Exception(
        'DataSet only support {}/{}'.format(support_dict, module_name))
    assert mode in ['Train', 'Eval',
                    'Test'], 'Mode should be Train, Eval or Test.'

    dataset = eval(module_name)(config, mode, logger, seed, epoch=epoch)
    loader_config = config[mode]['loader']
    batch_size = loader_config['batch_size_per_card']
    drop_last = loader_config['drop_last']
    shuffle = loader_config['shuffle']
    num_workers = loader_config['num_workers']
    if 'pin_memory' in loader_config.keys():
        pin_memory = loader_config['use_shared_memory']
    else:
        pin_memory = False

    sampler = None
    batch_sampler = None
    if 'sampler' in config[mode]:
        config_sampler = config[mode]['sampler']
        sampler_name = config_sampler.pop('name')
        batch_sampler = eval(sampler_name)(dataset, **config_sampler)
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
    if len(data_loader) == 0:
        logger.error(
            f'No Images in {mode.lower()} dataloader, please ensure\n'
            '\t1. The images num in the train label_file_list should be larger than or equal with batch size.\n'
            '\t2. The annotation file and path in the configuration file are provided normally.\n'
            '\t3. The BatchSize is large than images.')
        sys.exit()
    return data_loader
