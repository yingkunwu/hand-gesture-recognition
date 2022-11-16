import torch.nn as nn


class MultiTasksLoss(nn.Module):
    def __init__(self):
        super(MultiTasksLoss, self).__init__()
        self.regression = nn.MSELoss()
        self.classification = nn.CrossEntropyLoss()

    def forward(self, heatmap_pred, label_pred, heatmap_target, label_target):
        if heatmap_pred is not None and label_pred is not None:
            regression_loss = self.regression(heatmap_pred, heatmap_target)
            classification_lass = self.classification(label_pred, label_target)
            return regression_loss + classification_lass * 0.001
        elif heatmap_pred is not None:
            regression_loss = self.regression(heatmap_pred, heatmap_target)
            return regression_loss
        else:
            classification_lass = self.classification(label_pred, label_target)
            return classification_lass