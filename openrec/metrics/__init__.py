import copy

__all__ = ['build_metric']

support_dict = [
    'RecMetric', 'RecMetricLong', 'RecGTCMetric', 'RecMPGMetric', 'CMERMetric'
]


def build_metric(config):
    config = copy.deepcopy(config)
    module_name = config.pop('name')
    assert module_name in support_dict, Exception(
        'metric only support {}'.format(support_dict))

    # Lazy import
    if module_name == 'RecMetric':
        from .rec_metric import RecMetric
        module_class = RecMetric(**config)
    elif module_name == 'RecGTCMetric':
        from .rec_metric_gtc import RecGTCMetric
        module_class = RecGTCMetric(**config)
    elif module_name == 'RecMetricLong':
        from .rec_metric_long import RecMetricLong
        module_class = RecMetricLong(**config)
    elif module_name == 'RecMPGMetric':
        from .rec_metric_mgp import RecMPGMetric
        module_class = RecMPGMetric(**config)
    elif module_name == 'CMERMetric':
        from .rec_metric_cmer import CMERMetric
        module_class = CMERMetric(**config)

    return module_class
