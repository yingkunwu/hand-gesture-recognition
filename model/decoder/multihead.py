import torch
from torch import nn


class MultiHeadDecoder(nn.Module):
    def __init__(self, num_joints, num_classes):
        super(MultiHeadDecoder, self).__init__()

        self.deconv1 = self._make_deconv_layer(
            512, 256, kernel_size=4, stride=2, padding=1)
        self.deconv2 = self._make_deconv_layer(
            512, 256, kernel_size=4, stride=2, padding=1)
        self.deconv3 = self._make_deconv_layer(
            512, 256, kernel_size=4, stride=2, padding=1)

        self.bridge1 = self._make_bridge(256, 256)
        self.bridge2 = self._make_bridge(256, 256)
        self.bridge3 = self._make_bridge(256, 256)

        self.bridge4 = nn.Sequential(
            nn.Conv2d(
                in_channels=num_joints,
                out_channels=num_joints,
                kernel_size=1,
                stride=1,
                padding=0),
            nn.BatchNorm2d(num_joints, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.heatmap_layer = nn.Conv2d(
            in_channels=512,
            out_channels=num_joints,
            kernel_size=1,
            stride=1,
            padding=0
        )

        self.classification_layer = nn.Sequential(
            nn.Linear(1856, 400),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(400, num_classes)
        )

    def _make_deconv_layer(self, c_in, c_out, kernel_size, stride, padding):
        layers = []
        layers.append(
            nn.ConvTranspose2d(
                in_channels=c_in,
                out_channels=c_out,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=0,
                bias=False))
        layers.append(nn.BatchNorm2d(c_out, momentum=0.1))
        layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    def _make_bridge(self, c_in, c_out):
        layers = []
        layers.append(
            nn.Conv2d(
                in_channels=c_in,
                out_channels=c_out,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False))
        layers.append(nn.BatchNorm2d(c_out, momentum=0.1))
        layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    def forward(self, x):
        h = self.deconv1(x)
        h1 = self.bridge1(h)
        h = torch.cat((h, h1), dim=1)

        h = self.deconv2(h)
        h2 = self.bridge2(h)
        h = torch.cat((h, h2), dim=1)

        h = self.deconv3(h)
        h3 = self.bridge3(h)
        h = torch.cat((h, h3), dim=1)

        h = self.heatmap_layer(h)
        heatmap = h

        # prepare backbone features for classification
        features = self.avgpool(x)
        features = features.view(features.size(0), -1)

        # prepare heatmap features for classification
        h1 = self.avgpool(h1)
        h1 = h1.view(h1.size(0), -1)

        h2 = self.avgpool(h2)
        h2 = h2.view(h2.size(0), -1)

        h3 = self.avgpool(h3)
        h3 = h3.view(h3.size(0), -1)

        h4 = self.bridge4(h)
        h4 = torch.mean(h4, dim=1, keepdim=True)
        h4 = h4.view(h4.size(0), -1)

        # concatenate all features for classification
        y = torch.cat((features, h1, h2, h3, h4), dim=1)
        label = self.classification_layer(y)

        return label, heatmap

    def init_weights(self):
        for m in self.deconv1.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        for m in self.deconv2.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        for m in self.deconv3.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
