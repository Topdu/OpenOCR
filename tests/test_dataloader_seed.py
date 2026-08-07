import random
from types import SimpleNamespace

from tools.data import build_dataloader


def _config(label_file, seed):
    return {
        'Global': {
            'seed': seed,
            'distributed': False,
        },
        'Train': {
            'dataset': {
                'name': 'SimpleDataSet',
                'data_dir': str(label_file.parent),
                'label_file_list': [str(label_file)],
                'ratio_list': [0.5],
                'transforms': [],
            },
            'loader': {
                'batch_size_per_card': 1,
                'drop_last': False,
                'num_workers': 0,
                'pin_memory': False,
                'shuffle': True,
            },
        },
    }


def _data_order(label_file, process_seed, explicit_seed=None):
    logger = SimpleNamespace(info=lambda *args: None, error=lambda *args: None)
    random.seed(process_seed)
    loader = build_dataloader(
        _config(label_file, process_seed),
        'Train',
        logger,
        seed=explicit_seed,
        task='det',
    )
    return loader.dataset.data_lines


def test_simple_dataset_preserves_process_and_explicit_seeds(tmp_path):
    label_file = tmp_path / 'labels.txt'
    label_file.write_text(
        ''.join(f'image-{i}.jpg\t[]\n' for i in range(40)),
        encoding='utf-8',
    )

    seed_48_first = _data_order(label_file, 48)
    seed_48_second = _data_order(label_file, 48)
    seed_49 = _data_order(label_file, 49)
    explicit_7_after_48 = _data_order(label_file, 48, explicit_seed=7)
    explicit_7_after_49 = _data_order(label_file, 49, explicit_seed=7)

    assert seed_48_first == seed_48_second
    assert seed_48_first != seed_49
    assert explicit_7_after_48 == explicit_7_after_49
