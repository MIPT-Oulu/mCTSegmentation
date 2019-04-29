from torch import nn
import torch
import torch.nn.functional as F


class BCEWithLogitsLoss2d(nn.Module):
    """Computationally stable version of 2D BCE loss

    """

    def __init__(self):
        super(BCEWithLogitsLoss2d, self).__init__()

        self.bce_loss = nn.BCEWithLogitsLoss(None, reduction='mean')

    def forward(self, logits, targets):
        logits_flat = logits.view(-1)
        targets_flat = targets.view(-1)
        return self.bce_loss(logits_flat, targets_flat)


class SoftJaccardLoss(nn.Module):
    """SoftJaccard loss

    """

    def __init__(self, use_log=False):
        super(SoftJaccardLoss, self).__init__()
        self.use_log = use_log

    def forward(self, logits, labels):
        num = labels.size(0)
        m1 = torch.sigmoid(logits.view(num, -1))
        m2 = labels.view(num, -1)
        intersection = (m1 * m2).sum(1)
        score = (intersection + 1e-15) / (m1.sum(1) + m2.sum(1) - intersection + 1e-15)
        jaccard = score.sum(0) / num

        if not self.use_log:
            score = 1 - jaccard
        else:
            score = -torch.log(jaccard)
        return score


class CombinedLoss(nn.Module):
    """Combination loss.

    Used to combine several existing losses, e.g. Dice and BCE

    """

    def __init__(self, losses, weights=None):
        super(CombinedLoss, self).__init__()
        self.losses = losses
        if weights is None:
            weights = [1 / len(losses)] * len(losses)

        self.weights = weights

    def forward(self, inputs, targets):
        loss = 0
        for l, w in zip(self.losses, self.weights):
            loss += l(inputs, targets) * w
        return loss


class FocalLoss(nn.Module):
    """
    Focal loss. Based on the implementation by BloodAxe:
    https://github.com/BloodAxe/pytorch-toolbelt/blob/develop/pytorch_toolbelt/losses/functional.py#L8

    """
    def __init__(self, gamma=2.0, alpha=0.25, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, targets):
        num = logits.size(0)
        logits = logits.view(num, -1)
        targets = targets.view(num, -1)
        logpt = -F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(logpt)

        # compute the loss
        loss = -((1 - pt).pow(self.gamma)) * logpt

        if self.alpha is not None:
            loss = loss * (self.alpha * targets + (1 - self.alpha) * (1 - targets))

        if self.reduction == 'mean':
            loss = loss.mean()
        if self.reduction == 'sum':
            loss = loss.sum()
        if self.reduction == 'batchwise_mean':
            loss = loss.sum(0)

        return loss