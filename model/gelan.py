import torch
from torch import nn


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [
            d * (x - 1) + 1 for x in k
        ]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [
            x // 2 for x in k
        ]  # auto-pad
    return p


class Conv(nn.Module):
    """
    Standard convolution block with batchnorm and activation
    """
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """
        Args:
            c1: input channels
            c2: output channels
            k: kernel size
            s: stride
            p: padding
            g: groups
            d: dilation
            act: activation
        """
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=c1,
            out_channels=c2,
            kernel_size=k,
            stride=s,
            padding=autopad(k, p, d),
            groups=g,
            dilation=d,
            bias=False)
        self.bn = nn.BatchNorm2d(c2)
        if act is True:
            self.act = self.default_act
        else:
            if isinstance(act, nn.Module):
                self.act = act
            else:
                self.act = nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ResBasicBlock(nn.Module):
    # ResNet basicblock
    def __init__(self, c1, c2, shortcut=True):
        """
        Args:
            c1: input channels
            c2: output channels
            shortcut: whether to use residual layers
        """
        super(ResBasicBlock, self).__init__()
        self.cv1 = Conv(c1, c2, 3, 1)
        self.cv2 = Conv(c2, c2, 3, 1, act=False)
        self.act = nn.SiLU()
        self.add = shortcut
        self.downsample = None

        if self.add and c1 != c2:
            self.downsample = Conv(c1, c2, 1, 1, act=False)

    def forward(self, x):
        residual = x
        if self.add:
            if self.downsample is not None:
                residual = self.downsample(residual)
            out = residual + self.cv2(self.cv1(x))
        else:
            out = self.cv2(self.cv1(x))

        return self.act(out)


class ResBottleneck(nn.Module):
    # ResNet bottleneck
    def __init__(self, c1, c2, shortcut=True, e=0.5):
        """
        Args:
            c1: input channels
            c2: output channels
            shortcut: whether to use residual layers
            e: bottleneck expansion factor
        """
        super(ResBottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c_, 3, 1)
        self.cv3 = Conv(c_, c2, 1, 1, act=False)
        self.act = nn.SiLU()
        self.add = shortcut and c1 == c2
        self.downsample = None

        if self.add and c1 != c2:
            self.downsample = Conv(c1, c2, 1, 1, act=False)

    def forward(self, x):
        residual = x
        if self.add:
            if self.downsample is not None:
                residual = self.downsample(residual)
            out = residual + self.cv3(self.cv2(self.cv1(x)))
        else:
            out = self.cv3(self.cv2(self.cv1(x)))

        return self.act(out)


class GELANBlock(nn.Module):
    def __init__(self, c_in, c_out, c_hid1, c_hid2, block, nblocks=1):
        super(GELANBlock, self).__init__()
        self.cv1 = Conv(c_in, c_hid1, 1, 1)
        self.cv2 = nn.Sequential(
            *([block(c_hid1//2, c_hid2)]
              + [block(c_hid2, c_hid2) for _ in range(nblocks-1)])
        )
        self.cv3 = nn.Sequential(
            *(block(c_hid2, c_hid2) for _ in range(nblocks))
        )
        self.cv4 = Conv(c_hid1+(2*c_hid2), c_out, 1, 1)

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        for m in [self.cv2, self.cv3]:
            y.append(m(y[-1]))

        return self.cv4(torch.cat(y, 1))


class GELANNet(nn.Module):
    def __init__(self, gelan_type):
        super(GELANNet, self).__init__()
        gelan_spec = {
            'small': [ResBasicBlock, [1, 1, 1, 1]],
            'large': [ResBasicBlock, [2, 2, 2, 2]]
        }

        block, layers = gelan_spec[gelan_type]

        self.conv1 = Conv(3, 64, 3, 2)
        self.conv2 = Conv(64, 128, 3, 2)
        self.cspelan1 = GELANBlock(128, 128, 128, 64, block, layers[0])
        self.down1 = Conv(128, 256, 3, 2)
        self.cspelan2 = GELANBlock(256, 256, 256, 128, block, layers[1])
        self.down2 = Conv(256, 512, 3, 2)
        self.cspelan3 = GELANBlock(512, 512, 512, 256, block, layers[2])
        # self.down3 = Conv(512, 512, 3, 2)
        # self.cspelan4 = GELANBlock(512, 512, 512, 256, block, layers[3])

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.cspelan1(x)
        x = self.down1(x)
        x = self.cspelan2(x)
        x = self.down2(x)
        x = self.cspelan3(x)
        # x = self.down3(x)
        # x = self.cspelan4(x)

        return x


if __name__ == '__main__':
    model = GELANNet()
    x = torch.randn(32, 3, 256, 256)
    out = model(x)
    print(out.shape)
