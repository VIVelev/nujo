import pytest

import nujo.objective as obj
from nujo.autodiff.tensor import Tensor
from nujo.init.random import rand


def test_l1_loss(input, target):
    loss_fn = obj.L1Loss()
    assert isinstance(loss_fn(input, target), Tensor)


def test_l2_loss(input, target):
    loss_fn = obj.L2Loss()
    assert isinstance(loss_fn(input, target), Tensor)


def test_binary_cross_entropy(input, target):
    loss_fn = obj.BinaryCrossEntropy()
    assert isinstance(loss_fn(input, target), Tensor)


def test_cross_entropy(input, target):
    loss_fn = obj.CrossEntropy()
    assert isinstance(loss_fn(input, target), Tensor)


@pytest.fixture
def input():
    return rand(100, 42)


@pytest.fixture
def target():
    return rand(100, 42)
