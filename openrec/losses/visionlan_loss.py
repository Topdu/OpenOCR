import torch
import torch.nn.functional as F
from torch import nn


def flatten_label(target):
    label_flatten = []
    label_length = []
    for i in range(0, target.size()[0]):
        cur_label = target[i].tolist()
        label_flatten += cur_label[:cur_label.index(0) + 1]
        label_length.append(cur_label.index(0) + 1)
    label_flatten = torch.LongTensor(label_flatten)
    label_length = torch.IntTensor(label_length)
    return (label_flatten, label_length)


def _flatten(sources, lengths):
    return torch.cat([t[:l] for t, l in zip(sources, lengths)])


class VisionLANLoss(nn.Module):

    def __init__(self,
                 training_step='LA',
                 ratio_res=0.5,
                 ratio_sub=0.5,
                 **kwargs):
        super(VisionLANLoss, self).__init__()
        self.loss_func = nn.CrossEntropyLoss(reduction='mean')
        self.ratio_res = ratio_res
        self.ratio_sub = ratio_sub
        assert training_step in ['LF_1', 'LF_2', 'LA']
        self.training_step = training_step

    def forward(self, pred, batch):
        text_pre, text_rem, text_mas, _ = pred
        target = batch[1].to(dtype=torch.int64)
        label_flatten, length = flatten_label(target)
        text_pre = _flatten(text_pre, length)
        if self.training_step == 'LF_1':
            loss = self.loss_func(text_pre, label_flatten.to(text_pre.device))
        else:
            target_res = batch[2].to(dtype=torch.int64)
            target_sub = batch[3].to(dtype=torch.int64)
            label_flatten_res, length_res = flatten_label(target_res)
            label_flatten_sub, length_sub = flatten_label(target_sub)
            text_rem = _flatten(text_rem, length_res)
            text_mas = _flatten(text_mas, length_sub)
            loss_ori = self.loss_func(text_pre,
                                      label_flatten.to(text_pre.device))
            loss_res = self.loss_func(text_rem,
                                      label_flatten_res.to(text_rem.device))
            loss_mas = self.loss_func(text_mas,
                                      label_flatten_sub.to(text_mas.device))
            loss = loss_ori + loss_res * self.ratio_res + loss_mas * self.ratio_sub

        return {'loss': loss}
