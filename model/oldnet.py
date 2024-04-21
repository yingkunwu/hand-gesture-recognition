from torch import nn

from .encoder.resnet import ResNet
from .decoder.multihead import MultiHeadDecoder


class OldNet(nn.Module):
    def __init__(self, num_joints, num_classes, feature_size):
        super(OldNet, self).__init__()

        self.encoder = ResNet(18)
        self.decoder = MultiHeadDecoder(num_joints, num_classes)

    def forward(self, x):
        features = self.encoder(x)
        label, heatmaps = self.decoder(features)

        return label, heatmaps, None


if __name__ == '__main__':
    import torch
    from thop import profile
    from thop import clever_format

    name = "gelans"
    model = OldNet(21, 19, 12)
    x = torch.randn(1, 3, 192, 192)
    label, heatmap, _ = model(x)
    print(label.size(), heatmap.size())

    flops, params = profile(model, inputs=(x,))
    flops, params = clever_format([flops, params], "%.3f")
    print(f"Model: {name}, FLOPs: {flops}, Params: {params}")
