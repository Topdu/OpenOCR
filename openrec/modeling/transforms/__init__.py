__all__ = ['build_transform']


def build_transform(config):
    from .tps import TPS
    from .stn import STN_ON
    from .tsrn import TSRN
    from .tbsrn import TBSRN
    from .gaspin_transformer import GA_SPIN_Transformer as GA_SPIN

    support_dict = ['TPS', 'STN_ON', 'GA_SPIN', 'TSRN', 'TBSRN']

    module_name = config.pop('name')
    assert module_name in support_dict, Exception(
        'transform only support {}'.format(support_dict))
    module_class = eval(module_name)(**config)
    return module_class
