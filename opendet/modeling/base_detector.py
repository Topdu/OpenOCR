import torch
from torch import nn

from opendet.modeling.backbones import build_backbone
from opendet.modeling.necks import build_neck
from opendet.modeling.heads import build_head

__all__ = ['BaseDetector']


class BaseDetector(nn.Module):

    def __init__(self, config):
        """the module for OCR.

        args:
            config (dict): the super parameters for module.
        """
        super(BaseDetector, self).__init__()
        in_channels = config.get('in_channels', 3)
        self.use_wd = config.get('use_wd', True)

        # build backbone
        if 'Backbone' not in config or config['Backbone'] is None:
            self.use_backbone = False
        else:
            self.use_backbone = True
            config['Backbone']['in_channels'] = in_channels
            self.backbone = build_backbone(config['Backbone'])
            in_channels = self.backbone.out_channels

        # build neck
        if 'Neck' not in config or config['Neck'] is None:
            self.use_neck = False
        else:
            self.use_neck = True
            config['Neck']['in_channels'] = in_channels
            self.neck = build_neck(config['Neck'])
            in_channels = self.neck.out_channels

        # build head
        if 'Head' not in config or config['Head'] is None:
            self.use_head = False
        else:
            self.use_head = True
            config['Head']['in_channels'] = in_channels
            self.head = build_head(config['Head'])

    @torch.jit.ignore
    def no_weight_decay(self):
        if self.use_wd:
            if hasattr(self.backbone, 'no_weight_decay'):
                no_weight_decay = self.backbone.no_weight_decay()
            else:
                no_weight_decay = {}
            if hasattr(self.head, 'no_weight_decay'):
                no_weight_decay.update(self.head.no_weight_decay())
            return no_weight_decay
        else:
            return {}

    def forward(self, x, data=None):
        if self.use_backbone:
            x = self.backbone(x)
        if self.use_neck:
            x = self.neck(x)
        if self.use_head:
            x = self.head(x, data=data)
        return x
