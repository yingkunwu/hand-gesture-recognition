import torch
import torch.nn as nn
import torch.nn.functional as F


class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(conv_block, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=11, padding=5),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=11, stride=1, padding=5),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=11, stride=1, padding=5),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(128, out_channels, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        x = self.block(x)
        return x

class ConvolutionalPoseMachine(nn.Module):
    def __init__(self, n_joints):
        super(ConvolutionalPoseMachine, self).__init__()
        self.share_features = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=9, stride=1, padding=4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(128, 128, kernel_size=9, stride=1, padding=4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(128, 128, kernel_size=9, stride=1, padding=4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(128, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU()
        )
        self.block1 = nn.Sequential(
            nn.Conv2d(32, 512, kernel_size=9, stride=1, padding=4),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(512, n_joints + 1, kernel_size=1, stride=1, padding=0)
        )
        self.block2 = conv_block(32 + n_joints + 1, n_joints + 1)
        self.block3 = conv_block(32 + n_joints + 1, n_joints + 1)
        self.block4 = conv_block(32 + n_joints + 1, n_joints + 1)

    def _stage1(self, x):
        x = self.block1(x)
        return x

    def _stage2(self, x, heatmap):
        x = torch.cat([x, heatmap], dim=1)
        x = self.block2(x)
        return x

    def _stage3(self, x, heatmap):
        x = torch.cat([x, heatmap], dim=1)
        x = self.block3(x)
        return x

    def _stage4(self, x, heatmap):
        x = torch.cat([x, heatmap], dim=1)
        x = self.block4(x)
        return x

    def forward(self, x):
        share = self.share_features(x)

        heatmap1 = self._stage1(share)
        heatmap2 = self._stage2(share, heatmap1)
        heatmap3 = self._stage3(share, heatmap2)
        heatmap4 = self._stage4(share, heatmap3)
        return heatmap1, heatmap2, heatmap3, heatmap4
