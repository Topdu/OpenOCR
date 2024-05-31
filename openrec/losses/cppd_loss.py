import torch
import torch.nn.functional as F
from torch import nn


class CPPDLoss(nn.Module):

    def __init__(self,
                 smoothing=False,
                 ignore_index=100,
                 pos_len=False,
                 sideloss_weight=1.0,
                 max_len=25,
                 **kwargs):
        super(CPPDLoss, self).__init__()
        self.edge_ce = nn.CrossEntropyLoss(reduction='mean',
                                           ignore_index=ignore_index)
        self.char_node_ce = nn.CrossEntropyLoss(reduction='mean')
        if pos_len:
            self.pos_node_ce = nn.CrossEntropyLoss(reduction='mean',
                                                   ignore_index=ignore_index)
        else:
            self.pos_node_ce = nn.BCEWithLogitsLoss(reduction='mean')

        self.smoothing = smoothing
        self.ignore_index = ignore_index
        self.pos_len = pos_len
        self.sideloss_weight = sideloss_weight
        self.max_len = max_len + 1

    def label_smoothing_ce(self, preds, targets):
        zeros_ = torch.zeros_like(targets)
        ignore_index_ = zeros_ + self.ignore_index
        non_pad_mask = torch.not_equal(targets, ignore_index_)

        tgts = torch.where(targets == ignore_index_, zeros_, targets)
        eps = 0.1
        n_class = preds.shape[1]
        one_hot = F.one_hot(tgts, preds.shape[1])
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(preds, dim=1)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).mean()
        return loss

    def forward(self, pred, batch):
        node_feats, edge_feats = pred
        node_tgt = batch[2]
        char_tgt = batch[1]

        # updated code
        char_num_label = torch.clip(node_tgt[:, :-self.max_len].flatten(0, 1),
                                    0, node_feats[0].shape[-1] - 1)
        loss_char_node = self.char_node_ce(node_feats[0].flatten(0, 1),
                                           char_num_label)
        if self.pos_len:
            loss_pos_node = self.pos_node_ce(
                node_feats[1].flatten(0, 1),
                node_tgt[:, -self.max_len:].flatten(0, 1))
        else:
            loss_pos_node = self.pos_node_ce(
                node_feats[1].flatten(0, 1),
                node_tgt[:, -self.max_len:].flatten(0, 1).float())
        loss_node = loss_char_node + loss_pos_node
        # -----
        edge_feats = edge_feats.flatten(0, 1)
        char_tgt = char_tgt.flatten(0, 1)
        if self.smoothing:
            loss_edge = self.label_smoothing_ce(edge_feats, char_tgt)
        else:
            loss_edge = self.edge_ce(edge_feats, char_tgt)

        return {
            'loss': self.sideloss_weight * loss_node + loss_edge,
            'loss_node': self.sideloss_weight * loss_node,
            'loss_edge': loss_edge,
        }
