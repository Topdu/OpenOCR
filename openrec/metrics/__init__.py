import copy

__all__ = ['build_metric']

from .rec_metric import RecMetric
from .rec_metric_gtc import RecGTCMetric
from .rec_metric_long import RecMetricLong
from .rec_metric_mgp import RecMPGMetric

support_dict = ['RecMetric', 'RecMetricLong', 'RecGTCMetric', 'RecMPGMetric']


def build_metric(config):
    config = copy.deepcopy(config)
    module_name = config.pop('name')
    assert module_name in support_dict, Exception(
        'metric only support {}'.format(support_dict))
    module_class = eval(module_name)(**config)
    return module_class
