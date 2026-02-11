"""
This code is based on https://github.com/D1noFuzi/cocoamapping/.
Kalischek, N., Lang, N., Renier, C. et al. Cocoa plantations are associated with deforestation in Côte d’Ivoire and Ghana.
Nat Food 4, 384–393 (2023). https://doi.org/10.1038/s43016-023-00751-8

This is the xception network slightly adapted from
https://github.com/tstandley/Xception-PyTorch/blob/master/xception.py
and
https://github.com/hoya012/pytorch-Xception/blob/master/Xception_pytorch.ipynb

It is based on
Francois Chollet
Xception: Deep Learning with Depthwise Separable Convolutions
https://arxiv.org/pdf/1610.02357.pdf
"""
import torch.nn as nn


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1):
        super(SeparableConv2d, self).__init__()

        self.depthwise = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, dilation=dilation, groups=in_channels, bias=False)

        self.pointwise = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1,
                                   padding=0, dilation=1, groups=1, bias=False)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class PointwiseBlock(nn.Module):

    def __init__(self, in_channels, filters):
        super(PointwiseBlock, self).__init__()

        self.in_channels = in_channels
        self.filters = filters

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=filters[0], kernel_size=1, stride=1, bias=True)
        self.bn1 = nn.BatchNorm2d(filters[0])

        self.conv2 = nn.Conv2d(in_channels=filters[0], out_channels=filters[1], kernel_size=1, stride=1, bias=True)
        self.bn2 = nn.BatchNorm2d(filters[1])

        self.conv3 = nn.Conv2d(in_channels=filters[1], out_channels=filters[2], kernel_size=1, stride=1, bias=True)
        self.bn3 = nn.BatchNorm2d(filters[2])

        self.relu = nn.ReLU(inplace=True)
        if in_channels != filters[-1]:
            self.conv_shortcut = nn.Conv2d(in_channels=in_channels, out_channels=filters[2], kernel_size=1, stride=1, bias=True)
            self.bn_shortcut = nn.BatchNorm2d(filters[2])

    def forward(self, x):
        if self.in_channels == self.filters[-1]:
            # identity shortcut
            shortcut = x
        else:
            shortcut = self.conv_shortcut(x)
            shortcut = self.bn_shortcut(shortcut)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = out + shortcut
        out = self.relu(out)

        return out


class SepConvBlock(nn.Module):

    def __init__(self, in_channels, filters):
        super(SepConvBlock, self).__init__()

        self.in_channels = in_channels
        self.filters = filters

        self.sepconv1 = SeparableConv2d(in_channels=in_channels, out_channels=filters[0], kernel_size=3)
        self.bn1 = nn.BatchNorm2d(filters[0])

        self.sepconv2 = SeparableConv2d(in_channels=filters[0], out_channels=filters[1], kernel_size=3)
        self.bn2 = nn.BatchNorm2d(filters[1])

        self.relu = nn.ReLU(inplace=False)
        if in_channels != filters[1]:
            self.conv_shortcut = nn.Conv2d(in_channels=in_channels, out_channels=filters[1], kernel_size=1, stride=1, bias=True)
            self.bn_shortcut = nn.BatchNorm2d(filters[1])

    def forward(self, x):
        if self.in_channels == self.filters[-1]:
            # identity shortcut
            shortcut = x
        else:
            shortcut = self.conv_shortcut(x)
            shortcut = self.bn_shortcut(shortcut)

        out = self.relu(x)
        out = self.sepconv1(out)
        out = self.bn1(out)

        out = self.relu(out)
        out = self.sepconv2(out)
        out = self.bn2(out)

        out = out + shortcut

        return out
