from math import e

import pytest

from nujo.autodiff.tensor import Tensor
from nujo.nn.activations import BinaryStep, Sigmoid, TanH


def test_binary_step(input_value):
    # Test Forward pass
    output = BinaryStep()(input_value)
    assert output == 0

    # Test Backward pass
    output.backward()
    assert input_value.grad != 0


def test_sigmoid(input_value):
    # Test Forward pass
    output = Sigmoid()(input_value)

    x = input_value.value
    assert output == 1 / (1 + e**-x)

    # Test Backward pass
    output.backward()
    assert input_value.grad == output * (1 - output)


def test_tanh(input_value):
    # Test Forward pass
    output = TanH()(input_value)

    x = input_value.value
    assert output == (e**x - e**-x) / (e**x + e**-x)

    # Test Backward pass
    output.backward()
    assert input_value.grad == 1 - output**2


@pytest.fixture
def input_value():
    return Tensor(0.42, name='test')
