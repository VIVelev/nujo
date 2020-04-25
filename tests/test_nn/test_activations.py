from math import e

import pytest
from numpy import equal, isclose, maximum

import nujo.nn.activations as activ
from nujo.autodiff.tensor import Tensor

# ====================================================================================================
# Test BinaryStep activation function


def test_binary_step(input_value):
    # Test Forward pass
    output = activ.BinaryStep()(input_value)
    assert equal(output.value, [[0, 0], [1, 0]]).all()

    # Test Backward pass
    output.backward()
    assert equal(input_value.grad, 0).all()


# ====================================================================================================
# Test Sigmoid activation function


def test_sigmoid(input_value):
    # Test Forward pass
    output = activ.Sigmoid()(input_value)

    x = input_value.value
    assert equal(output.value, 1 / (1 + e**-x)).all()

    # Test Backward pass
    output.backward()
    assert equal(input_value.grad, output.value * (1 - output.value)).all()


# ====================================================================================================
# Test TanH activation function


def test_tanh(input_value):
    # Test Forward pass
    output = activ.TanH()(input_value)

    x = input_value.value
    assert isclose(output.value, (e**x - e**-x) / (e**x + e**-x)).all()

    # Test Backward pass
    output.backward()
    assert equal(input_value.grad, 1 - output.value**2).all()


# ====================================================================================================
# Test ReLU activation function


def test_relu(input_value):
    # Test Forward pass
    output = activ.ReLU()(input_value)

    x = input_value.value
    assert equal(output.value, maximum(0, x)).all()

    # Test Backward pass
    output.backward()
    assert equal(input_value.grad[input_value.grad > 0], 1).all()
    assert equal(input_value.grad[input_value.grad <= 0], 0).all()


# ====================================================================================================
# Test LeakyReLU activation function


def test_leaky_relu(input_value):
    # Test Forward pass
    eps = 0.1
    output = activ.LeakyReLU(eps=eps)(input_value)

    x = input_value.value
    assert equal(output.value, maximum(eps * x, x)).all()

    # Test Backward pass
    output.backward()
    assert equal(input_value.grad[input_value.grad > 0], 1).all()
    assert equal(input_value.grad[input_value.grad <= 0], eps).all()


# ====================================================================================================
# Test Swish activation function


def test_swish(input_value):
    # Test Forward pass
    beta = 1
    output = activ.Swish(beta=beta)(input_value)

    x = input_value.value
    sigma = activ.Sigmoid()(beta * x).value
    assert equal(output.value, x * sigma).all()

    # Test Backward pass
    output.backward()
    assert equal(input_value.grad,
                 output.value + sigma * (1 - output.value)).all()


# ====================================================================================================
# Test Softmax activation function


def test_softmax(input_value):
    # Test Forward pass
    output = activ.Softmax()(input_value)

    assert output.shape == (2, 3)

    # Test Backward pass
    assert output.backward()


# ====================================================================================================
# Fixtures


@pytest.fixture
def input_value():
    return Tensor([[0.42, 0.32, 0.34], [0.6, 0.1, 1.1]], name='test_input')


# ====================================================================================================
