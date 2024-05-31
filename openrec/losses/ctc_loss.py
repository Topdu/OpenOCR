import paddle
from paddle import nn


class CTCLoss(nn.Layer):
    def __init__(self, use_focal_loss=False, **kwargs):
        super(CTCLoss, self).__init__()
        self.loss_func = nn.CTCLoss(blank=0, reduction='none')
        self.use_focal_loss = use_focal_loss

    def forward(self, predicts, batch):
        if isinstance(predicts, (list, tuple)):
            predicts = predicts[-1]
        predicts = predicts.transpose((1, 0, 2))
        N, B, _ = predicts.shape
        preds_lengths = paddle.to_tensor(
            [N] * B, dtype='int64', place=paddle.CPUPlace())
        labels = batch[1].astype("int32")
        label_lengths = batch[2].astype('int64')
        loss = self.loss_func(predicts, labels, preds_lengths, label_lengths)
        if self.use_focal_loss:
            weight = paddle.exp(-loss)
            weight = paddle.subtract(paddle.to_tensor([1.0]), weight)
            weight = paddle.square(weight)
            loss = paddle.multiply(loss, weight)
        loss = loss.mean()
        return {'loss': loss}
