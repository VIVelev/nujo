import pytest

from nujo.autodiff.tensor import Tensor
from nujo.init.random import rand
from nujo.objective.quantitative import L1Loss, L2Loss


def test_l1_loss(input, target):
    loss_fn = L1Loss()
    loss = loss_fn(input, target)

    assert isinstance(loss, Tensor)
    assert loss.shape == ()


def test_l2_loss(input, target):
    loss_fn = L2Loss()
    loss = loss_fn(input, target)

    assert isinstance(loss, Tensor)
    assert loss.shape == ()


@pytest.fixture
def input():
    return rand(100, 1)


@pytest.fixture
def target():
    return rand(100, 1)
