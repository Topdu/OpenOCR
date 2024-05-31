import torch.nn.functional as F
from torch import nn


class LPVLoss(nn.Module):

    def __init__(self, label_smoothing=0.0, **kwargs):
        super(LPVLoss, self).__init__()
        self.label_smoothing = label_smoothing

    def forward(self, preds, batch):
        max_len = batch[2].max()
        tgt = batch[1][:, 1:2 + max_len]

        tgt = tgt.flatten(0, 1)
        loss = 0
        loss_dict = {}
        for i, pred in enumerate(preds):
            pred = pred.flatten(0, 1)
            loss_i = F.cross_entropy(
                pred,
                tgt,
                reduction='mean',
                label_smoothing=self.label_smoothing,
                ignore_index=pred.shape[1] + 1,
            )  # self.loss_func(pred, tgt)
            loss += loss_i
            loss_dict['loss' + str(i)] = loss_i
        loss_dict['loss'] = loss
        return loss_dict
