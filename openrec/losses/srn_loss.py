import torch.nn.functional as F
from torch import nn


class SRNLoss(nn.Module):

    def __init__(self, label_smoothing=0.0, **kwargs):
        super(SRNLoss, self).__init__()
        self.label_smoothing = label_smoothing

    def forward(self, preds, batch):
        pvam_preds, gsrm_preds, vsfd_preds = preds

        label = batch[1].reshape([-1])

        ignore_index = pvam_preds.shape[-1] + 1

        loss_pvam = F.cross_entropy(pvam_preds,
                                    label,
                                    reduction='mean',
                                    label_smoothing=self.label_smoothing,
                                    ignore_index=ignore_index)
        loss_gsrm = F.cross_entropy(gsrm_preds,
                                    label,
                                    reduction='mean',
                                    label_smoothing=self.label_smoothing,
                                    ignore_index=ignore_index)
        loss_vsfd = F.cross_entropy(vsfd_preds,
                                    label,
                                    reduction='mean',
                                    label_smoothing=self.label_smoothing,
                                    ignore_index=ignore_index)

        loss_dict = {}
        loss_dict['loss_pvam'] = loss_pvam
        loss_dict['loss_gsrm'] = loss_gsrm
        loss_dict['loss_vsfd'] = loss_vsfd

        loss_dict['loss'] = loss_pvam * 3.0 + loss_gsrm * 0.15 + loss_vsfd
        return loss_dict
