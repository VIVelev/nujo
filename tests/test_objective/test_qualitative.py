import pytest

from nujo.autodiff.tensor import Tensor
from nujo.init.random import rand
from nujo.objective.qualitative import BinaryCrossEntropy, CrossEntropy

# ====================================================================================================
# Test Binary Cross Entropy


def test_binary_cross_entropy(inputs, targets):
    loss_fn = BinaryCrossEntropy()
    loss = loss_fn(inputs, targets)

    assert isinstance(loss, Tensor)
    assert loss.shape == (1, 1)


# ====================================================================================================
# Test Cross Entropy


def test_cross_entropy(inputs, targets):
    loss_fn = CrossEntropy()
    loss = loss_fn(inputs, targets)

    assert isinstance(loss, Tensor)
    assert loss.shape == (1, 1)


# ====================================================================================================
# Unit Test fixtures


@pytest.fixture
def inputs():
    return rand(42, 100)


@pytest.fixture
def targets():
    return rand(42, 100)


# ====================================================================================================
