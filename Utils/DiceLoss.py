import torch
import torch.nn.functional as F

def dice_loss(preds, targets):
    smooth = 1e-6

    preds = preds.contiguous().view(preds.size(0), -1)
    targets = targets.contiguous().view(targets.size(0), -1)

    intersection = (preds * targets).sum(dim=1)
    union = preds.sum(dim=1) + targets.sum(dim=1)

    dice_score = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice_score.mean()
