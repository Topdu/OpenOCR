from torch import nn


class LISTERLoss(nn.Module):

    def __init__(self, **kwargs):
        super(LISTERLoss, self).__init__()

    def forward(self, predicts, batch):
        # predicts = predicts['res']
        # loss = predicts
        if isinstance(predicts, list):
            predicts = predicts[0]
        return predicts
