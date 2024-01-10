import os
import torch
from torch import nn
from collections import OrderedDict

from .encoder import ResNet
from .decoder import PoseDecoder


class PoseResNet(nn.Module):
    def __init__(self, num_layers, num_joints, num_classes):
        super(PoseResNet, self).__init__()

        self.encoder = ResNet(num_layers)
        self.decoder = PoseDecoder(num_joints, num_classes)

    def forward(self, x):
        features = self.encoder(x)
        heatmaps = self.decoder(features)

        return heatmaps

    def init_weights(self, pretrained=''):
        if os.path.isfile(pretrained):
            self.decoder.init_weights()

            # load pretrained model
            checkpoint = torch.load(pretrained)

            # only load weights from encoders
            state_dict = OrderedDict()
            for key in checkpoint.keys():
                if key.startswith('encoder'):
                    state_dict[key] = checkpoint[key]
            self.load_state_dict(state_dict, strict=False)
        else:
            raise ValueError("Pretrained model '{}' does not exist."
                             .format(pretrained))
