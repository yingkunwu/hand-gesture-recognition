import torch.nn as nn
import torch.nn.functional as F


def ce_loss(pred, target):
    criterion = nn.BCELoss(reduction='sum')
    print(pred.shape, target.shape)
    x0_h, x0_w = target[0].size(-2), target[0].size(-1)
    print(x0_h, x0_w)
    pred = F.interpolate(pred, size=(x0_h, x0_w), mode='bilinear', align_corners=True)
    print(pred.shape)
    loss = criterion(pred, target)
    return loss


def mse_loss(pred, target):
    criterion = nn.MSELoss(reduction='sum')
    print(pred.shape, target.shape)
    x0_h, x0_w = target[0].size(-2), target[0].size(-1)
    print(x0_h, x0_w)
    pred = F.interpolate(pred, size=(x0_h, x0_w), mode='bilinear', align_corners=True)
    print(pred.shape)
    loss = criterion(pred, target)
    return loss
