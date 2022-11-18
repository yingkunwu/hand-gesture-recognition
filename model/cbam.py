import torch
import torch.nn as nn

class channel_attention_module(nn.Module):
    def __init__(self, ch, ratio=8):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Linear(ch, ch//ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch//ratio, ch, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.avg_pool(x).squeeze(-1).squeeze(-1)
        x1 = self.mlp(x1)

        x2 = self.max_pool(x).squeeze(-1).squeeze(-1)
        x2 = self.mlp(x2)

        feats = x1 + x1
        feats = self.sigmoid(feats).unsqueeze(-1).unsqueeze(-1)
        refined_feats = x * feats

        return refined_feats


class spatial_attention_module(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Conv2d(2, 1, kernel_size=5, stride=1, padding=2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = torch.mean(x, dim=1, keepdim=True)
        x2, _ = torch.max(x, dim=1, keepdim=True)

        feats = torch.cat([x1, x2], dim=1)
        feats = self.conv(feats)
        feats = self.sigmoid(feats)
        refined_feats = x * feats

        return refined_feats

class cbam(nn.Module):
    def __init__(self, conv_channel):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=conv_channel, out_channels=conv_channel, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True)
        )

        self.ca = channel_attention_module(conv_channel)
        self.sa = spatial_attention_module()

    def forward(self, x):
        residual = x
        out = self.conv(x)
        out = self.ca(out)
        out = self.sa(out)
        out += residual
        return out


if __name__ == "__main__":
    x = torch.randn((8, 32, 128, 128))
    module = cbam(32)
    y = module(x)
    print(y.shape)
