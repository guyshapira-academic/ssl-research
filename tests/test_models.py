"""Unit tests for ssl_research.models subpackage.
"""
import pytest
import torch
from ssl_research.models import cnn


@pytest.mark.parametrize("in_channels", [16])
@pytest.mark.parametrize("out_channels", [8, 16, 32])
@pytest.mark.parametrize("skip_connection", [False, True])
@pytest.mark.parametrize("stride", [1, 2, 3])
def test_conv_block(
    in_channels: int, out_channels: int, skip_connection: bool, stride: int
) -> None:
    conv_block = cnn.ConvBlock(
        in_channels, out_channels, skip_connection=skip_connection, stride=stride
    )

    x = torch.rand(7, in_channels, 96, 96)
    y = conv_block(x)

    assert y.shape == (7, out_channels, 96 // stride, 96 // stride)
