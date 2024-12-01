__all__ = ['build_backbone']


def build_backbone(config):
    # det backbone
    from .repvit import RepSVTR_det

    support_dict = ['RepSVTR_det']

    module_name = config.pop('name')
    assert module_name in support_dict, Exception(
        'head only support {}'.format(support_dict))
    module_class = eval(module_name)(**config)
    return module_class
