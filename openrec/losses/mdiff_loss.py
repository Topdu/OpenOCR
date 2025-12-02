from torch import nn


class MDiffLoss(nn.Module):

    def __init__(self, **kwargs):
        super(MDiffLoss, self).__init__()

    def forward(self, predicts, batch):

        return {'loss': predicts}
