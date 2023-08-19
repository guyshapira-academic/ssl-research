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


@pytest.mark.parametrize("width", [(16, 32, 64, 128)])
@pytest.mark.parametrize("depth", [(2, 2, 2, 2)])
@pytest.mark.parametrize("skip_connections", [False, True])
@pytest.mark.parametrize("output_dim", [128, 256])
def test_cnn(
    width: tuple, depth: tuple, skip_connections: bool, output_dim: int
) -> None:
    cnn_model = cnn.CNN(
        width=width,
        depth=depth,
        skip_connections=skip_connections,
        output_dim=output_dim,
    )

    x = torch.rand(7, 3, 96, 96)
    y = cnn_model(x)

    assert y.shape == (7, output_dim)


def test_feature_forward() -> None:
    cnn_model = cnn.CNN(
        width=(16, 32, 64, 128),
        depth=(2, 2, 2, 2),
        skip_connections=False,
        output_dim=256,
    )

    x = torch.rand(7, 3, 96, 96)

    y = x
    for level in cnn_model.features_forward(x):
        y = level
        assert isinstance(y, torch.Tensor)

    assert y.shape == (7, 256)
