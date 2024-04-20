import os
import torch
from torch import nn
from collections import OrderedDict

from .encoder.gelan import GELANNet
from .encoder.resnet import ResNet
from .decoder.multihead import MultiHeadDecoder
from .decoder.transformer import ViT


class MultiTaskNet(nn.Module):
    def __init__(self, num_layers, num_joints, num_classes):
        super(MultiTaskNet, self).__init__()

        self.encoder = GELANNet("small")
        # self.encoder = ResNet(18)
        # self.decoder = MultiHeadDecoder(num_joints, num_classes)
        self.proj = nn.Conv2d(512, 256, 1, bias=False)
        self.decoder = ViT(
            num_classes=num_classes,
            num_joints=num_joints,
            feature_size=12,
            dim=256, depth=4,
            heads=8,
            head_dim=256,
            mlp_dim=512)

    def forward(self, x):
        features = self.encoder(x)
        features = self.proj(features)
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


if __name__ == '__main__':
    from thop import profile
    from thop import clever_format

    name = "gelans"
    model = MultiTaskNet(name, 21, 19)
    x = torch.randn(1, 3, 192, 192)
    label, heatmap = model(x)
    print(label.size(), heatmap.size())

    flops, params = profile(model, inputs=(x,))
    flops, params = clever_format([flops, params], "%.3f")
    print(f"Model: {name}, FLOPs: {flops}, Params: {params}")
