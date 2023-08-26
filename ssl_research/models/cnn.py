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


class CNN(nn.Module):
    """Full CNN model.

    Parameters:
        width (tuple): Number of channels for each spatial resolution.
        depth (tuple): Number of convolutional blocks per spatial resolution.
        skip_connections (bool): Whether to add residual connections.
            Default: False.
    """

    def __init__(
        self,
        width: tuple = (16, 32, 64, 128),
        depth: tuple = (2, 2, 2, 2),
        output_dim: int = 256,
        skip_connections: bool = False,
    ):
        super().__init__()

        self.output_dim = output_dim
        self.num_features = len(width) + 1

        self.conv_in = nn.Conv2d(
            in_channels=3, out_channels=width[0], kernel_size=3, stride=1, padding=1
        )

        self.levels = nn.ModuleList()
        for i in range(len(width)):
            level = nn.Sequential()
            for j in range(depth[i]):
                if j == 0 and i != 0:
                    level.append(
                        ConvBlock(
                            width[i - 1],
                            width[i],
                            skip_connection=skip_connections,
                            stride=2,
                        )
                    )
                else:
                    level.append(
                        ConvBlock(
                            width[i],
                            width[i],
                            skip_connection=skip_connections,
                            stride=1,
                        )
                    )
            self.levels.append(level)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(width[-1], output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv_in(x)

        for level in self.levels:
            out = level(out)

        out = self.global_pool(out)
        out = out.view(out.shape[0], -1)
        out = self.fc(out)

        return out

    def features_forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv_in(x)

        for level in self.levels:
            out = level(out)
            yield out

        out = self.global_pool(out)
        out = out.view(out.shape[0], -1)
        out = self.fc(out)

        yield out


def resnet20(output_dim: int) -> CNN:
    """ResNet-20 model.

    Parameters:
        output_dim (int): Dimension of the output vector.
    """
    return CNN(
        width=(16, 32, 64),
        depth=(3, 3, 3),
        output_dim=output_dim,
        skip_connections=True,
    )


def vanilla20(output_dim: int) -> CNN:
    """ResNet-20 model.

    Parameters:
        output_dim (int): Dimension of the output vector.
    """
    return CNN(
        width=(16, 32, 64),
        depth=(2, 2, 2),
        output_dim=output_dim,
        skip_connections=False,
    )
