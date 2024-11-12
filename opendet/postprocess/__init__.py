import copy

__all__ = ['build_post_process']

from .db_postprocess import DBPostProcess

support_dict = ['DBPostProcess']


def build_post_process(config, global_config=None):
    config = copy.deepcopy(config)
    module_name = config.pop('name')
    if global_config is not None:
        config.update(global_config)
    assert module_name in support_dict, Exception(
        'det post process only support {}'.format(support_dict))
    module_class = eval(module_name)(**config)
    return module_class
