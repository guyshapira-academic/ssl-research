"""Unit tests for ssl_research.ncc submodule
"""
import numpy as np
import pytest
import torch
from ssl_research import ncc


@pytest.mark.parametrize("test_data", [True, False])
@pytest.mark.parametrize("tensor_data", [True, False])
def test_ncc_accuracy(test_data: bool, tensor_data: bool):
    X_train = np.random.randn(100, 10)
    y_train = np.random.randint(0, 10, 100)

    if test_data:
        X_test = np.random.randn(100, 10)
        y_test = np.random.randint(0, 10, 100)
    else:
        X_test = None
        y_test = None

    if tensor_data:
        X_train = torch.from_numpy(X_train)
        y_train = torch.from_numpy(y_train)
        if test_data:
            X_test = torch.from_numpy(X_test)
            y_test = torch.from_numpy(y_test)

    acc = ncc.ncc_accuracy(X_train, y_train, X_test, y_test)
    assert isinstance(acc, float)
    assert 0.0 <= acc <= 1.0
