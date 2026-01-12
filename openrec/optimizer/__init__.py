import copy

import torch
from torch import nn
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
__all__ = ['build_optimizer']


def param_groups_weight_decay(model: nn.Module,
                              weight_decay=1e-5,
                              no_weight_decay_list=()):
    no_weight_decay_list = set(no_weight_decay_list)
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if param.ndim <= 1 or name.endswith(
                '.bias') or any(nd in name for nd in no_weight_decay_list):
            no_decay.append(param)
        else:
            decay.append(param)

    return [
        {
            'params': no_decay,
            'weight_decay': 0.0
        },
        {
            'params': decay,
            'weight_decay': weight_decay
        },
    ]


def build_optimizer(optim_config, lr_scheduler_config, epochs, step_each_epoch,
                    model):
    from . import lr

    config = copy.deepcopy(optim_config)

    if isinstance(model, nn.Module):
        # a model was passed in, extract parameters and add weight decays to appropriate layers
        weight_decay = config.get('weight_decay', 0.0)
        filter_bias_and_bn = (config.pop('filter_bias_and_bn')
                              if 'filter_bias_and_bn' in config else False)
        if weight_decay > 0.0 and filter_bias_and_bn:
            no_weight_decay = {}
            if hasattr(model, 'no_weight_decay'):
                no_weight_decay = model.no_weight_decay()
            parameters = param_groups_weight_decay(model, weight_decay,
                                                   no_weight_decay)
            config['weight_decay'] = 0.0
            # print('debug adamw')
        else:
            parameters = model.parameters()
    else:
        # iterable of parameters or param groups passed in
        parameters = model

    optim = getattr(torch.optim, config.pop('name'))(params=parameters,
                                                     **config)

    lr_config = copy.deepcopy(lr_scheduler_config)
    scheduler_name = lr_config.pop('name')
    
    total_steps = epochs * step_each_epoch

    if scheduler_name == "WarmupCosine":
        warmup_steps = lr_config.get('warmup_steps', 0)
        eta_min = lr_config.get('eta_min', 0.0)
        
        schedulers = []
        milestones = []
        
        # 1. Warmup 阶段
        if warmup_steps > 0:
            warmup_scheduler = LinearLR(
                optim,
                start_factor=1e-7, 
                end_factor=1.0,
                total_iters=warmup_steps
            )
            schedulers.append(warmup_scheduler)
            milestones.append(warmup_steps)
        
        remain_steps = max(1, total_steps - warmup_steps)
        
        T_max = remain_steps 

        cosine_scheduler = CosineAnnealingLR(
            optim,
            T_max=T_max,
            eta_min=eta_min
        )
        schedulers.append(cosine_scheduler)
        
        if len(schedulers) == 1:
            lr_scheduler = schedulers[0]
        else:
            lr_scheduler = SequentialLR(optim, schedulers=schedulers, milestones=milestones)
            
    else:
        from . import lr
        
        lr_config.update({
            'epochs': epochs,
            'step_each_epoch': step_each_epoch,
            'lr': config['lr']
        })
        lr_scheduler = getattr(lr, scheduler_name)(**lr_config)(optimizer=optim)
    
    return optim, lr_scheduler
