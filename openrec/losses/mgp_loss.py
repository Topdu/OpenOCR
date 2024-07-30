from torch import nn


class MGPLoss(nn.Module):

    def __init__(self, only_char=False, **kwargs):
        super(MGPLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss(reduction='mean', ignore_index=0)
        self.only_char = only_char

    def forward(self, pred, batch):
        if self.only_char:
            char_feats = pred
            char_tgt = batch[1].flatten(0, 1)
            char_loss = self.ce(char_feats.flatten(0, 1), char_tgt)
            return {'loss': char_loss}
        else:
            return self.forward_all(pred, batch)

    def forward_all(self, pred, batch):
        char_feats, dpe_feats, wp_feats = pred
        char_tgt = batch[1].flatten(0, 1)
        dpe_tgt = batch[2].flatten(0, 1)
        wp_tgt = batch[3].flatten(0, 1)
        char_loss = self.ce(char_feats.flatten(0, 1), char_tgt)
        dpe_loss = self.ce(dpe_feats.flatten(0, 1), dpe_tgt)
        wp_loss = self.ce(wp_feats.flatten(0, 1), wp_tgt)
        loss = char_loss + dpe_loss + wp_loss
        return {
            'loss': loss,
            'char_loss': char_loss,
            'dpe_loss': dpe_loss,
            'wp_loss': wp_loss
        }
