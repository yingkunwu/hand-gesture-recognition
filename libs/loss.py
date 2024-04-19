import torch.nn as nn


class JointsMSELoss(nn.Module):
    def __init__(self, use_target_weight):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape(
            (batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape(
            (batch_size, num_joints, -1)).split(1, 1)
        loss = 0

        for i in range(num_joints):
            heatmap_pred = heatmaps_pred[i].squeeze()
            heatmap_gt = heatmaps_gt[i].squeeze()
            if self.use_target_weight:
                loss += 0.5 * self.criterion(
                    heatmap_pred * target_weight[:, i],
                    heatmap_gt * target_weight[:, i]
                )
            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints


class MultiTaskLoss(nn.Module):
    def __init__(self, use_target_weight):
        super(MultiTaskLoss, self).__init__()
        self.joints_loss = JointsMSELoss(use_target_weight)
        self.class_loss = nn.CrossEntropyLoss()

    def forward(self, heatmap_pred, label_pred, heatmap_target, label_target):
        joints_loss = self.joints_loss(heatmap_pred, heatmap_target)
        class_loss = self.class_loss(label_pred, label_target)
        return joints_loss + class_loss * 0.002
