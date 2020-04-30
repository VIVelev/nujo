import pytest
import torch
from numpy import allclose, random

import nujo as nj


def test_scalar_diff(scalar_tensors):
    (X_nj, y_nj, W1_nj, W2_nj, X_torch, y_torch, W1_torch,
     W2_torch) = scalar_tensors

    loss_nj = nj.mean((X_nj * W1_nj * W2_nj - y_nj)**2)
    loss_torch = torch.mean((X_torch * W1_torch * W2_torch - y_torch)**2)

    # Test Forward
    assert allclose(loss_nj.value, loss_torch.detach().numpy())

    # Test Backward
    loss_nj.backward()
    loss_torch.backward()

    assert allclose(W1_nj.grad.value, W1_torch.grad.detach().numpy())
    assert allclose(W2_nj.grad.value, W2_torch.grad.detach().numpy())


def test_matrix_diff(matrix_tensors):
    (X_nj, y_nj, W1_nj, W2_nj, X_torch, y_torch, W1_torch,
     W2_torch) = matrix_tensors

    loss_nj = nj.mean((X_nj @ W1_nj @ W2_nj - y_nj)**2)
    loss_torch = torch.mean((X_torch @ W1_torch @ W2_torch - y_torch)**2)

    # Test Forward
    assert allclose(loss_nj.value, loss_torch.detach().numpy())

    # Test Backward
    loss_nj.backward()
    loss_torch.backward()

    assert allclose(W1_nj.grad.value, W1_torch.grad.detach().numpy())
    assert allclose(W2_nj.grad.value, W2_torch.grad.detach().numpy())


@pytest.fixture
def scalar_tensors():
    X = random.rand()
    y = random.rand()

    W1 = random.rand()
    W2 = random.rand()

    X_nj = nj.Tensor(X)
    y_nj = nj.Tensor(y)
    W1_nj = nj.Tensor(W1, diff=True, name='W1')
    W2_nj = nj.Tensor(W2, diff=True, name='W2')

    X_torch = torch.tensor(X)
    y_torch = torch.tensor(y)
    W1_torch = torch.tensor(W1, requires_grad=True)
    W2_torch = torch.tensor(W2, requires_grad=True)

    return X_nj, y_nj, W1_nj, W2_nj, X_torch, y_torch, W1_torch, W2_torch


@pytest.fixture
def matrix_tensors():
    X = random.rand(3, 3)
    y = random.rand(3, 1)

    W1 = random.rand(3, 2)
    W2 = random.rand(2, 1)

    X_nj = nj.Tensor(X)
    y_nj = nj.Tensor(y)
    W1_nj = nj.Tensor(W1, diff=True, name='W1')
    W2_nj = nj.Tensor(W2, diff=True, name='W2')

    X_torch = torch.tensor(X)
    y_torch = torch.tensor(y)
    W1_torch = torch.tensor(W1, requires_grad=True)
    W2_torch = torch.tensor(W2, requires_grad=True)

    return X_nj, y_nj, W1_nj, W2_nj, X_torch, y_torch, W1_torch, W2_torch
