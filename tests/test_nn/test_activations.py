import pytest
from numpy import allclose, exp, maximum, sum

import nujo.nn.activations as activ
from nujo.autodiff.tensor import Tensor

# ====================================================================================================
# Test BinaryStep activation function


def test_binary_step(inputs):
    # Test Forward pass
    output = activ.BinaryStep()(inputs)
    assert (output == [[0, 0, 0], [1, 0, 1]]).all()

    # Test Backward pass
    output.backward()
    assert (inputs.grad == 0).all()


# ====================================================================================================
# Test Sigmoid activation function


def test_sigmoid(inputs):
    # Test Forward pass
    output = activ.Sigmoid()(inputs)

    x = inputs.value
    assert (output == 1 / (1 + exp(-x))).all()

    # Test Backward pass
    output.backward()
    assert (inputs.grad == output.value * (1 - output.value)).all()


# ====================================================================================================
# Test TanH activation function


def test_tanh(inputs):
    # Test Forward pass
    output = activ.TanH()(inputs)

    x = inputs.value
    assert allclose(output.value, (exp(x) - exp(-x)) / (exp(x) + exp(-x)))

    # Test Backward pass
    output.backward()
    assert (inputs.grad == 1 - output.value**2).all()


# ====================================================================================================
# Test ReLU activation function


def test_relu(inputs):
    # Test Forward pass
    output = activ.ReLU()(inputs)

    x = inputs.value
    assert (output == maximum(0, x)).all()

    # Test Backward pass
    output.backward()
    assert (inputs.grad[inputs.grad > 0] == 1).all()
    assert (inputs.grad[inputs.grad <= 0] == 0).all()


# ====================================================================================================
# Test LeakyReLU activation function


def test_leaky_relu(inputs):
    # Test Forward pass
    eps = 0.1
    output = activ.LeakyReLU(eps=eps)(inputs)

    x = inputs.value
    assert (output == maximum(eps * x, x)).all()

    # Test Backward pass
    output.backward()
    assert (inputs.grad[inputs.grad > 0] == 1).all()
    assert (inputs.grad[inputs.grad <= 0] == eps).all()


# ====================================================================================================
# Test Swish activation function


def test_swish(inputs):
    # Test Forward pass
    beta = 1
    output = activ.Swish(beta=beta)(inputs)

    x = inputs.value
    sigma = activ.Sigmoid()(beta * x).value
    assert (output == x * sigma).all()

    # Test Backward pass
    output.backward()
    assert (inputs.grad == output.value + sigma * (1 - output.value)).all()


# ====================================================================================================
# Test Softmax activation function


def test_softmax(inputs):
    # Test Forward pass
    output = activ.Softmax()(inputs)

    exps = exp(inputs.value)
    sums = sum(exps, axis=0, keepdims=True)
    assert allclose(output.value, exps / sums)

    # Test Backward pass
    # TODO: Test Backward pass appropriately.
    output.backward()
    assert True


# ====================================================================================================
# Unit Test fixtures


@pytest.fixture
def inputs():
    return Tensor([[0.42, 0.32, 0.34], [0.6, 0.1, 1.1]], diff=True)


# ====================================================================================================
