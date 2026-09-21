"""
Script-Aware MoE Loss for Multilingual Text Recognition.

This loss combines:
1. Standard AR cross-entropy loss (for text recognition)
2. Script classification loss (optional, when script labels are available)
"""

import torch.nn.functional as F
from torch import nn


class ScriptMoELoss(nn.Module):
    """Combined loss for Script-Aware MoE Decoder.

    Args:
        label_smoothing (float): Label smoothing for AR cross-entropy loss.
        script_cls_weight (float): Weight for script classification loss.
            Supervision signal for the script classifier head.
            Typical values: 0.1 ~ 1.0
        ignore_index (int): Index to ignore in AR loss (unused, kept for compat).
    """

    def __init__(
        self,
        label_smoothing=0.1,
        script_cls_weight=0.5,
        ignore_index=0,
        **kwargs,
    ):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.script_cls_weight = script_cls_weight

    def compute_ar_loss(self, pred, batch):
        """Standard autoregressive cross-entropy loss.

        Args:
            pred: [B, T, vocab_size-2] prediction logits
            batch: list of [images, labels, lengths, ...]

        Returns:
            loss: scalar tensor
        """
        max_len = batch[2].max()
        tgt = batch[1][:, 1:2 + max_len]
        pred = pred.flatten(0, 1)
        tgt = tgt.reshape([-1])
        loss = F.cross_entropy(
            pred, tgt,
            reduction='mean',
            label_smoothing=self.label_smoothing,
            ignore_index=pred.shape[1] + 1,
        )
        return loss

    def compute_script_cls_loss(self, script_logits, script_targets):
        """Script classification loss (cross-entropy).

        Provides explicit supervision for the script classifier head,
        which helps the router learn meaningful script-aware representations.

        Args:
            script_logits: [B, num_experts] predicted script distribution
            script_targets: [B] ground-truth script indices

        Returns:
            loss: scalar tensor
        """
        return F.cross_entropy(script_logits, script_targets, reduction='mean')

    def forward(self, preds, batch):
        """Compute total loss.

        Args:
            preds: dict from ScriptMoEDecoder.forward() containing:
                - 'logit': [B, T, vocab_size-2]
                - 'script_logits': [B, num_experts]
                - 'script_targets': [B] (optional)
            batch: list of [images, labels, lengths, ...]

        Returns:
            dict with individual losses and total loss
        """
        result = {}

        # 1. AR cross-entropy loss
        ar_loss = self.compute_ar_loss(preds['logit'], batch)
        result['ar_loss'] = ar_loss
        total_loss = ar_loss

        # 2. Script classification loss (if script labels are available)
        if 'script_targets' in preds and preds['script_targets'] is not None:
            script_loss = self.compute_script_cls_loss(
                preds['script_logits'], preds['script_targets']
            )
            result['script_loss'] = script_loss
            total_loss = total_loss + self.script_cls_weight * script_loss

        result['loss'] = total_loss
        return result
