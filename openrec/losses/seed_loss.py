import torch.nn.functional as F
from torch import nn
import torch


class CosineEmbeddingLoss(nn.Module):

    def __init__(self, margin=0.0):
        super(CosineEmbeddingLoss, self).__init__()
        self.margin = margin
        self.epsilon = 1e-12

    def forward(self, x1, x2):
        similarity = torch.sum(x1 * x2, axis=-1) / (
            torch.norm(x1, dim=-1) * torch.norm(x2, dim=-1) + self.epsilon)
        return (1 - similarity).mean()


class SEEDLoss(nn.Module):

    def __init__(self, label_smoothing=0.1, ignore_index=0, **kwargs):
        super(SEEDLoss, self).__init__()
        self.label_smoothing = label_smoothing
        self.loss_sem = CosineEmbeddingLoss()

    def forward(self, preds, batch):
        embedding_vectors, pred = preds
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
        sem_target = batch[3].float()

        sem_loss = torch.sum(self.loss_sem(embedding_vectors, sem_target))
        return {
            'loss': loss + 0.1 * sem_loss,
            'rec_loss': loss,
            'sem_loss': sem_loss
        }
