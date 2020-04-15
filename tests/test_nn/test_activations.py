import pytest

from nujo.autodiff.tensor import Tensor
from nujo.nn.activations import BinaryStep, Sigmoid, TanH


def test_binary_step(input_value):
    output = BinaryStep()(input_value)
    assert isinstance(output, Tensor)
    output.backward()


def test_sigmoid(input_value):
    output = Sigmoid()(input_value)
    assert isinstance(output, Tensor)
    output.backward()


def test_tanh(input_value):
    output = TanH()(input_value)
    assert isinstance(output, Tensor)
    output.backward()


@pytest.fixture
def input_value():
    return 42
