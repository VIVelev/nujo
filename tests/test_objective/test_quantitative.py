import pytest

from nujo.autodiff.tensor import Tensor
from nujo.init.random import rand
from nujo.objective.quantitative import L1Loss, L2Loss

# ====================================================================================================
# Test L1 Loss


def test_l1_loss(inputs, targets):
    loss_fn = L1Loss()
    loss = loss_fn(inputs, targets)

    assert isinstance(loss, Tensor)
    assert loss.shape == (1, 1)


# ====================================================================================================
# Test L2 Loss


def test_l2_loss(inputs, targets):
    loss_fn = L2Loss()
    loss = loss_fn(inputs, targets)

    assert isinstance(loss, Tensor)
    assert loss.shape == (1, 1)


# ====================================================================================================
# Unit Test fixtures


@pytest.fixture
def inputs():
    return rand(100, 1)


@pytest.fixture
def targets():
    return rand(100, 1)


# ====================================================================================================
