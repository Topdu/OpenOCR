import torch
import torch.nn.functional as F

from .ar_loss import ARLoss


def BanlanceMultiClassCrossEntropyLoss(x_o, x_t):
    # [B, num_cls, H, W]
    B, num_cls, H, W = x_o.shape
    x_o = x_o.reshape(B, num_cls, H * W).permute(0, 2, 1)
    # [B, H, W, num_cls]
    # generate gt
    x_t[x_t > 0.5] = 1
    x_t[x_t <= 0.5] = 0
    fg_x_t = x_t.sum(-1)  # [B, H, W]
    x_t = x_t.argmax(-1)  # [B, H, W]
    x_t[fg_x_t == 0] = num_cls - 1  # background
    x_t = x_t.reshape(B, H * W)
    # loss
    weight = torch.ones((B, num_cls)).type_as(x_o)  # the weight of bg is 1.
    num_bg = (x_t == (num_cls - 1)).sum(-1)  # [B]
    weight[:, :-1] = (num_bg / (H * W - num_bg + 1e-5)).unsqueeze(-1).expand(
        -1, num_cls - 1)
    logit = F.log_softmax(x_o, dim=-1)  # [B, H*W, num_cls]
    logit = logit * weight.unsqueeze(1)
    loss = -logit.gather(2, x_t.unsqueeze(-1).long())
    return loss.mean()


class CAMLoss(ARLoss):

    def __init__(self, label_smoothing=0.1, loss_weight_binary=1.5, **kwargs):
        super(CAMLoss, self).__init__(label_smoothing=label_smoothing)
        self.label_smoothing = label_smoothing
        self.loss_weight_binary = loss_weight_binary

    def forward(self, pred, batch):
        binary_mask = batch[-1]
        rec_loss = super().forward(pred['rec_output'], batch[:-1])['loss']
        output = pred
        loss_binary = self.loss_weight_binary * BanlanceMultiClassCrossEntropyLoss(
            output['pred_binary'], binary_mask)

        return {
            'loss': rec_loss + loss_binary,
            'rec_loss': rec_loss,
            'loss_binary': loss_binary
        }
