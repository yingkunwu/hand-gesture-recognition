import torch
import torch.nn as nn

from model.HRNet import HRNet


class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(conv_block, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(128, out_channels, kernel_size=1, stride=1, padding=0)
        )
    def forward(self, x):
        x = self.block(x)
        return x

class ConvolutionalPoseMachine(nn.Module):
    def __init__(self, num_masks, num_heatmaps, c=18):
        super(ConvolutionalPoseMachine, self).__init__()
        self.num_masks = num_masks
        self.num_heatmaps = num_heatmaps

        self.hrnet = HRNet(c, 128)

        # ***** limb segmentation stage *****
        self.block1 = nn.Sequential(
            nn.Conv2d(128, 512, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(512, self.num_masks, kernel_size=1, stride=1, padding=0)
        )
        self.block2 = conv_block(128 + self.num_masks, self.num_masks)
        self.block3 = conv_block(128 + self.num_masks, self.num_masks)

        # ***** keypoint detection stage *****
        self.block4 = conv_block(128 + self.num_masks, self.num_heatmaps)
        self.block5 = conv_block(128 + self.num_masks + self.num_heatmaps, self.num_heatmaps)
        self.block6 = conv_block(128 + self.num_masks + self.num_heatmaps, self.num_heatmaps)


    def forward(self, image):
        # ***** backbone *****
        features = self.hrnet(image)

        # ***** limb segmentation stage *****
        stage1 = self.block1(features)
        stage2 = self.block2(torch.cat([features, stage1], dim=1))
        stage3 = self.block3(torch.cat([features, stage2], dim=1))

        # ***** keypoint detection stage *****
        stage4 = self.block4(torch.cat([features, stage3], dim=1))
        stage5 = self.block5(torch.cat([features, stage3, stage4], dim=1))
        stage6 = self.block6(torch.cat([features, stage3, stage5], dim=1))

        limbmasks = torch.cat([stage1, stage2, stage3], dim=1)
        keypoints = torch.cat([stage4, stage5, stage6], dim=1)

        # Add sigmoid for limb
        return limbmasks.sigmoid(), keypoints

if __name__ == '__main__':
    model = ConvolutionalPoseMachine(7, 21)
    y = model(torch.ones(1, 3, 384, 384))
    print(y[0].shape, y[1].shape)
    print(torch.min(y).item(), torch.mean(y).item(), torch.max(y).item())
