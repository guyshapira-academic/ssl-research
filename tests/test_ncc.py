"""Unit tests for ssl_research.ncc submodule
"""
import numpy as np
import pytest
import torch
from ssl_research import ncc


@pytest.mark.parametrize("tensor_data", [True, False])
def test_ncc_accuracy(test_data: bool, tensor_data: bool):
    X = np.random.randn(100, 10)
    y = np.random.randint(0, 10, 100)

    if tensor_data:
        X = torch.from_numpy(X)
        y = torch.from_numpy(y)

    acc = ncc.ncc_accuracy(X, y)
    assert isinstance(acc, float)
    assert 0.0 <= acc <= 1.0
