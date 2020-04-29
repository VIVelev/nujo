# TODO: Test against PyTorch implementation of activation functions

import pytest
from numpy import exp, isclose, maximum, sum

import nujo.nn.activations as activ
from nujo.autodiff.tensor import Tensor

# ====================================================================================================
# Test BinaryStep activation function


def test_binary_step(input_vector):
    # Test Forward pass
    output = activ.BinaryStep()(input_vector)
    assert (output == [[0, 0, 0], [1, 0, 1]]).all()

    # Test Backward pass
    output.backward()
    assert (input_vector.grad == 0).all()


# ====================================================================================================
# Test Sigmoid activation function


def test_sigmoid(input_vector):
    # Test Forward pass
    output = activ.Sigmoid()(input_vector)

    x = input_vector.value
    assert (output == 1 / (1 + exp(-x))).all()

    # Test Backward pass
    output.backward()
    assert (input_vector.grad == output.value * (1 - output.value)).all()


# ====================================================================================================
# Test TanH activation function


def test_tanh(input_vector):
    # Test Forward pass
    output = activ.TanH()(input_vector)

    x = input_vector.value
    assert isclose(output.value, (exp(x) - exp(-x)) / (exp(x) + exp(-x))).all()

    # Test Backward pass
    output.backward()
    assert (input_vector.grad == 1 - output.value**2).all()


# ====================================================================================================
# Test ReLU activation function


def test_relu(input_vector):
    # Test Forward pass
    output = activ.ReLU()(input_vector)

    x = input_vector.value
    assert (output == maximum(0, x)).all()

    # Test Backward pass
    output.backward()
    assert (input_vector.grad[input_vector.grad > 0] == 1).all()
    assert (input_vector.grad[input_vector.grad <= 0] == 0).all()


# ====================================================================================================
# Test LeakyReLU activation function


def test_leaky_relu(input_vector):
    # Test Forward pass
    eps = 0.1
    output = activ.LeakyReLU(eps=eps)(input_vector)

    x = input_vector.value
    assert (output == maximum(eps * x, x)).all()

    # Test Backward pass
    output.backward()
    assert (input_vector.grad[input_vector.grad > 0] == 1).all()
    assert (input_vector.grad[input_vector.grad <= 0] == eps).all()


# ====================================================================================================
# Test Swish activation function


def test_swish(input_vector):
    # Test Forward pass
    beta = 1
    output = activ.Swish(beta=beta)(input_vector)

    x = input_vector.value
    sigma = activ.Sigmoid()(beta * x).value
    assert (output == x * sigma).all()

    # Test Backward pass
    output.backward()
    assert (input_vector.grad == output.value + sigma *
            (1 - output.value)).all()


# ====================================================================================================
# Test Softmax activation function


def test_softmax(input_vector):
    # Test Forward pass
    print(input_vector)
    output = activ.Softmax()(input_vector)
    print(output)

    exps = exp(input_vector.value)
    sums = sum(exps, axis=0, keepdims=True)

    assert (output == exps / sums).all()

    # Test Backward pass
    output.backward()
    assert True


# ====================================================================================================
# Fixtures


@pytest.fixture
def input_vector():
    return Tensor([[0.42, 0.32, 0.34], [0.6, 0.1, 1.1]],
                  diff=True,
                  name='test_input')


# ====================================================================================================
