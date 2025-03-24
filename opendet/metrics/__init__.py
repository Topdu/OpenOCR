import copy

__all__ = ['build_metric']

from .det_metric import DetMetric

support_dict = ['DetMetric']


def build_metric(config):
    config = copy.deepcopy(config)
    module_name = config.pop('name')
    assert module_name in support_dict, Exception(
        'metric only support {}'.format(support_dict))
    module_class = eval(module_name)(**config)
    return module_class
