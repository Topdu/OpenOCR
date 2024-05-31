from torch import nn


class PARSeqLoss(nn.Module):

    def __init__(self, **kwargs):
        super(PARSeqLoss, self).__init__()

    def forward(self, predicts, batch):
        # predicts = predicts['res']
        loss, _ = predicts
        return {'loss': loss}
