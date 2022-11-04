import torch.nn as nn
import torch.nn.functional as F


def ce_loss(pred, target):
    criterion = nn.BCELoss(reduction='sum')
    x0_h, x0_w = target[0].size(-2), target[0].size(-1)
    pred = F.interpolate(pred, size=(x0_h, x0_w), mode='bilinear', align_corners=True)
    loss = criterion(pred, target)
    return loss / (pred.shape[0] * x0_h * x0_w)


def mse_loss(pred, target):
    criterion = nn.MSELoss(reduction='sum')
    x0_h, x0_w = target[0].size(-2), target[0].size(-1)
    pred = F.interpolate(pred, size=(x0_h, x0_w), mode='bilinear', align_corners=True)
    loss = criterion(pred, target)
    return loss / (pred.shape[0] * x0_h * x0_w)
