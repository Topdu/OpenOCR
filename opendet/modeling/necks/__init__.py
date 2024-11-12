__all__ = ['build_neck']


def build_neck(config):
    # det backbone
    from .db_fpn import RSEFPN

    support_dict = ['RSEFPN']

    module_name = config.pop('name')
    assert module_name in support_dict, Exception(
        'det neck only support {}'.format(support_dict))
    module_class = eval(module_name)(**config)
    return module_class
