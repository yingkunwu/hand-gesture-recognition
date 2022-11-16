import torch
from torch import nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, kernel_size, padding, stride=1, downsample=None, bn_momentum=0.1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel_size, stride=1, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=bn_momentum)

        downsample = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes, momentum=bn_momentum),
        )

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, bn_momentum=0.1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=bn_momentum)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=bn_momentum)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


# derived from https://github.com/leoxiaobin/deep-high-resolution-net.pytorch
class ResNet(nn.Module):
    def __init__(self, resnet_size=50, nof_joints=21, nof_classes=10, bn_momentum=0.1):
        super(ResNet, self).__init__()
        resnet_spec = {
            50: (Bottleneck, [3, 4, 6, 3]),
            101: (Bottleneck, [3, 4, 23, 3]),
            152: (Bottleneck, [3, 8, 36, 3])
        }
        assert resnet_size in resnet_spec.keys()
        block, layers = resnet_spec[resnet_size]

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        #self.cnn2 = nn.Sequential(
        #    nn.Conv2d(789, 2048, kernel_size=5, stride=1, padding=2, bias=False),
        #    nn.BatchNorm2d(2048, momentum=bn_momentum),
        #    nn.ReLU(inplace=True)
        #)

        self.layer1 = self._make_layer(block, [64, 64], [256, 64], layers[0], stride=1, bn_momentum=bn_momentum)
        self.layer2 = self._make_layer(block, [256, 128], [512, 128], layers[1], stride=2, bn_momentum=bn_momentum)
        self.layer3 = self._make_layer(block, [512, 256], [1024, 256], layers[2], stride=2, bn_momentum=bn_momentum)
        self.layer4 = self._make_layer(block, [1024, 512], [2048, 512], layers[3], stride=2, bn_momentum=bn_momentum)

        self.deconv1 = self._make_deconv_layer(2048, 256, kernel_size=4, stride=2, padding=1, bn_momentum=bn_momentum)
        self.deconv2 = self._make_deconv_layer(256, 256, kernel_size=4, stride=2, padding=1, bn_momentum=bn_momentum)
        self.deconv3 = self._make_deconv_layer(256, 256, kernel_size=4, stride=2, padding=1, bn_momentum=bn_momentum)

        #self.bridge1 = BasicBlock(256, 256, kernel_size=5, stride=1, padding=2)
        #self.bridge2 = BasicBlock(256, 256, kernel_size=5, stride=1, padding=2)
        #self.bridge3 = BasicBlock(256, 256, kernel_size=5, stride=1, padding=2)
        #self.bridge4 = BasicBlock(nof_joints, nof_joints, kernel_size=5, stride=1, padding=2)

        self.down1 = BasicBlock(nof_joints, 128, kernel_size=5, stride=2, padding=2)
        self.down2 = BasicBlock(128, 128, kernel_size=5, stride=2, padding=2)
        self.down3 = BasicBlock(128, 128, kernel_size=5, stride=2, padding=2)

        self.heatmap_layer = nn.Conv2d(256, nof_joints, kernel_size=1, stride=1, padding=0)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.classification_layer = nn.Sequential(
            nn.Linear(2176, 2176),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(2176, nof_classes)
        )

    def _make_layer(self, block, conv_planes, identity_planes, blocks, stride, bn_momentum):
        layers = []
        for i in range(blocks):
            if i == 0:
                # conv block
                downsample = nn.Sequential(
                    nn.Conv2d(conv_planes[0], conv_planes[1] * block.expansion,
                            kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(conv_planes[1] * block.expansion, momentum=bn_momentum),
                )
                layers.append(block(conv_planes[0], conv_planes[1], stride, downsample))
            else:
                # identity block
                layers.append(block(identity_planes[0], identity_planes[1], 1, downsample=None))

        return nn.Sequential(*layers)
    
    def _make_deconv_layer(self, in_channels, out_channels, kernel_size, stride, padding, bn_momentum):
        layers =[]
        layers.append(nn.ConvTranspose2d(in_channels, out_channels, 
                            kernel_size=kernel_size, stride=stride, padding=padding, output_padding=0, bias=False))
        layers.append(nn.BatchNorm2d(out_channels, momentum=bn_momentum))
        layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x, ground_truth_heatmap):
        x = self.cnn(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        #x1 = self.deconv1(x)
        #x2 = self.deconv2(x1)
        #x3 = self.deconv3(x2)

        #heatmap = self.heatmap_layer(x3)

        #x1_ = F.interpolate(x1, size=(64, 64), mode='bilinear', align_corners=True)
        #x2_ = F.interpolate(x2, size=(64, 64), mode='bilinear', align_corners=True)
        #x3_ = F.interpolate(x3, size=(64, 64), mode='bilinear', align_corners=True)

        #x1_ = self.bridge1(x1)
        #x1_ = self.avgpool(x1_)

        #x2_ = self.bridge2(x2)
        #x2_ = self.avgpool(x2_)

        #x3_ = self.bridge3(x3)
        #x3_ = self.avgpool(x3_)

        #heatmap_ = self.bridge4(heatmap)
        heatmap_ = self.down1(ground_truth_heatmap)
        heatmap_ = self.down2(heatmap_)
        heatmap_ = self.down3(heatmap_)
        heatmap_ = self.avgpool(heatmap_)

        x = self.avgpool(x)
        x_ = torch.cat((x, heatmap_), dim=1)

        #x_ = self.cnn2(x3)
        #x_ = self.avgpool(x_)
        x_ = x_.view(x_.size(0), -1)
        label = self.classification_layer(x_)

        return None, label

if __name__ == "__main__":
    x = torch.randn((8, 3, 256, 256))
    module = ResNet()
    y1, y2 = module(x)
    print(y1.shape, y2.shape)

