from torch import nn


class RobustScannerLoss(nn.Module):

    def __init__(self, **kwargs):
        super(RobustScannerLoss, self).__init__()
        ignore_index = kwargs.get('ignore_index', 38)
        self.loss_func = nn.CrossEntropyLoss(reduction='mean',
                                             ignore_index=ignore_index)

    def forward(self, pred, batch):
        pred = pred[:, :-1, :]

        label = batch[1][:, 1:].reshape([-1])

        inputs = pred.reshape([-1, pred.shape[2]])

        loss = self.loss_func(inputs, label)
        return {'loss': loss}
