from torch import nn


class SMTRLoss(nn.Module):

    def __init__(self, **kwargs):
        super(SMTRLoss, self).__init__()

    def forward(self, predicts, batch):
        if isinstance(predicts, list):
            predicts = predicts[0]
        return predicts
