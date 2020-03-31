import pytest

from nujo.autodiff.tensor import Tensor
from nujo.init.random import rand
from nujo.objective.qualitative import BinaryCrossEntropy, CrossEntropy


def test_binary_cross_entropy(input, target):
    loss_fn = BinaryCrossEntropy()
    loss = loss_fn(input, target)

    assert isinstance(loss, Tensor)
    assert loss.shape == ()


def test_cross_entropy(input, target):
    loss_fn = CrossEntropy()
    loss = loss_fn(input, target)

    assert isinstance(loss, Tensor)
    assert loss.shape == ()


@pytest.fixture
def input():
    return rand(42, 100)


@pytest.fixture
def target():
    return rand(42, 100)
