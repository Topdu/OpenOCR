import torch
from torch import nn


class ABINetLoss(nn.Module):

    def __init__(self, smoothing=False, ignore_index=100, **kwargs):
        super(ABINetLoss, self).__init__()
        if ignore_index >= 0:
            self.loss_func = nn.CrossEntropyLoss(reduction='mean',
                                                 ignore_index=ignore_index)
        else:
            self.loss_func = nn.CrossEntropyLoss(reduction='mean')
        self.smoothing = smoothing

    def forward(self, pred, batch):
        loss = {}
        loss_sum = []
        for name, logits in pred.items():
            if isinstance(logits, list):
                logit_num = len(logits)
                if logit_num > 0:
                    all_tgt = torch.cat([batch[1]] * logit_num, 0)
                    all_logits = torch.cat(logits, 0)
                    flt_logtis = all_logits.reshape([-1, all_logits.shape[2]])
                    flt_tgt = all_tgt.reshape([-1])
                else:
                    continue
            else:
                flt_logtis = logits.reshape([-1, logits.shape[2]])
                flt_tgt = batch[1].reshape([-1])

            loss[name + '_loss'] = self.loss_func(flt_logtis, flt_tgt)
            loss_sum.append(loss[name + '_loss'])
        loss['loss'] = sum(loss_sum)
        return loss
