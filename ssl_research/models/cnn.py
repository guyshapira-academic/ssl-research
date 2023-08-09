"""This module contains implementations of CNN models.
"""
import torch
import torch.nn as nn


class Downsample(nn.Module):
    """Simple downsampling module.

    Parameters:
        stride (int): Stride of the convolutional layer. Default: 2.
    """

    def __init__(self, stride: int = 2):
        super().__init__()

        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, :: self.stride, :: self.stride]


class ConvBlock(nn.Module):
    """Convolutional block with two convolution layers, activations, normalizations
    and optionally a residual connection.

    Based on the ResNet block from the paper:
        "Deep Residual Learning for Image Recognition"
        (https://arxiv.org/abs/1512.03385)

    Parameters:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        skip_connection (bool): Whether to add a residual connection. Default: False.
            If True, the shortcut connection will be parameter-free, that is,
            the input will be subsampled and zero-padded if necessary.
        stride (int): Stride of the first convolution layer. Default: 1.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        skip_connection: bool = False,
        stride: int = 1,
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

        self.use_skip_connection = skip_connection
        self.skip_connection_block = None
        if skip_connection and in_channels == out_channels and stride == 1:
            self.skip_connection_block = nn.Identity()
        elif skip_connection:
            self.skip_connection_block = nn.Sequential()
            if stride != 1:
                self.skip_connection_block.append(Downsample(stride=stride))
            if in_channels != out_channels:
                self.skip_connection_block.append(
                    nn.ZeroPad2d((0, 0, 0, 0, 0, out_channels - in_channels))
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.use_skip_connection:
            residual = self.skip_connection_block(x)
            out += residual

        out = self.relu2(out)

        return out
