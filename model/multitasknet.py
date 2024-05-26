import torch
from torch import nn

from .gelan import GELANNet
from .transformer import ViT


class MultiTaskNet(nn.Module):
    def __init__(self, num_joints, num_classes, image_size):
        super(MultiTaskNet, self).__init__()

        self.encoder = GELANNet("small")
        self.proj = nn.Conv2d(512, 256, 1, bias=False)
        self.decoder = ViT(
            num_classes=num_classes,
            num_joints=num_joints,
            feature_size=[image_size[0] // 16, image_size[1] // 16],
            dim=256,
            depth=4,
            heads=8,
            head_dim=32,
            mlp_dim=256)

    def forward(self, x):
        features = self.encoder(x)
        features = self.proj(features)
        cls_out, hmap_out, attnmap = self.decoder(features)

        return cls_out, hmap_out, attnmap


if __name__ == '__main__':
    from thop import profile
    from thop import clever_format

    name = "gelans"
    model = MultiTaskNet(21, 19, 12)
    x = torch.randn(1, 3, 192, 192)
    label, heatmap, _ = model(x)
    print(label.size(), heatmap.size())

    flops, params = profile(model, inputs=(x,))
    flops, params = clever_format([flops, params], "%.3f")
    print(f"Model: {name}, FLOPs: {flops}, Params: {params}")
