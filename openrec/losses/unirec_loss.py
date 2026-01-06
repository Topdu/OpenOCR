from torch import nn


class UniRecLoss(nn.Module):

    def __init__(self, label_smoothing=0.1, **kwargs):
        super(UniRecLoss, self).__init__()

    def forward(self, pred, batch):
        # loss, vision_loss, text_loss = pred.loss
        loss = {'loss': pred.loss}
        return loss
