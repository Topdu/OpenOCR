import copy
from importlib import import_module

name_to_module = {
    'DBLoss': '.db_loss',
}


def build_loss(config):
    config = copy.deepcopy(config)
    module_name = config.pop('name')
    assert module_name in name_to_module, Exception(
        'loss only support {}'.format(list(name_to_module.keys())))

    if module_name in globals():
        module_class = globals()[module_name]
    else:
        module_path = name_to_module[module_name]
        module = import_module(module_path, package=__package__)
        module_class = getattr(module, module_name)

    return module_class(**config)
