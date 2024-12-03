import torch.nn.functional as F
from torch import nn


class ARLoss(nn.Module):

    def __init__(self, label_smoothing=0.1, ignore_index=0, **kwargs):
        super(ARLoss, self).__init__()
        self.label_smoothing = label_smoothing

    def forward(self, pred, batch):
        max_len = batch[2].max()
        tgt = batch[1][:, 1:2 + max_len]
        pred = pred.flatten(0, 1)
        tgt = tgt.reshape([-1])
        loss = F.cross_entropy(
            pred,
            tgt,
            reduction='mean',
            label_smoothing=self.label_smoothing,
            ignore_index=pred.shape[1] + 1,
        )  # self.loss_func(pred, tgt)
        return {'loss': loss}
