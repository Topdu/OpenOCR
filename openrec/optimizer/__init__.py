import copy
import paddle

__all__ = ['build_optimizer']


def build_lr_scheduler(lr_config, epochs, step_each_epoch):
    from . import learning_rate
    lr_config.update({'epochs': epochs, 'step_each_epoch': step_each_epoch})
    lr_name = lr_config.pop('name', 'Const')
    lr = getattr(learning_rate, lr_name)(**lr_config)()
    return lr


def build_optimizer(config, epochs, step_each_epoch, model):
    from . import regularizer, optimizer
    config = copy.deepcopy(config)
    # step1 build lr
    lr = build_lr_scheduler(config.pop('lr'), epochs, step_each_epoch)

    # step2 build regularization
    if 'regularizer' in config and config['regularizer'] is not None:
        reg_config = config.pop('regularizer')
        reg_name = reg_config.pop('name')
        if not hasattr(regularizer, reg_name):
            reg_name += 'Decay'
        reg = getattr(regularizer, reg_name)(**reg_config)()
    elif 'weight_decay' in config:
        reg = config.pop('weight_decay')
    else:
        reg = None

    # step3 build optimizer
    optim_name = config.pop('name')
    if 'clip_norm' in config:
        clip_norm = config.pop('clip_norm')
        grad_clip = paddle.nn.ClipGradByNorm(clip_norm=clip_norm)
    elif 'clip_norm_global' in config:
        clip_norm = config.pop('clip_norm_global')
        grad_clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=clip_norm)
    else:
        grad_clip = None
    optim = getattr(optimizer, optim_name)(learning_rate=lr,
                                           weight_decay=reg,
                                           grad_clip=grad_clip,
                                           **config)
    return optim(model), lr
