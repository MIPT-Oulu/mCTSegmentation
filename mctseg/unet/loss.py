from torch import nn


class BCEWithLogitsLoss2d(nn.Module):
    """Computationally stable version of 2D BCE loss

    """

    def __init__(self, weight=None, size_average=True):
        super(BCEWithLogitsLoss2d, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss(weight, size_average)

    def forward(self, logits, targets):
        logits_flat = logits.view(-1)
        targets_flat = targets.view(-1)
        return self.bce_loss(logits_flat, targets_flat)


class BinaryDiceLoss(nn.Module):
    """SoftDice loss

    """

    def __init__(self):
        super(BinaryDiceLoss, self).__init__()

    def forward(self, logits, labels):
        num = labels.size(0)
        m1 = logits.view(num, -1)
        m2 = labels.view(num, -1)
        intersection = (m1 * m2)
        score = 2. * (intersection.sum(1) + 1e-15) / (m1.sum(1) + m2.sum(1) + 1e-15)
        score = 1 - score.sum() / num
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



