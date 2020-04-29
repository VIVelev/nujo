import pytest
from numpy import diag, exp, hstack, isclose, maximum, repeat, sum

import nujo.nn.activations as activ
from nujo.autodiff.tensor import Tensor

# ====================================================================================================
# Test BinaryStep activation function


def test_binary_step(input_value):
    # Test Forward pass
    output = activ.BinaryStep()(input_value)
    assert (output == [[0, 0, 0], [1, 0, 1]]).all()

    # Test Backward pass
    output.backward()
    assert (input_value.grad == 0).all()


# ====================================================================================================
# Test Sigmoid activation function


def test_sigmoid(input_value):
    # Test Forward pass
    output = activ.Sigmoid()(input_value)

    x = input_value.value
    assert (output == 1 / (1 + exp(-x))).all()

    # Test Backward pass
    output.backward()
    assert (input_value.grad == output.value * (1 - output.value)).all()


# ====================================================================================================
# Test TanH activation function


def test_tanh(input_value):
    # Test Forward pass
    output = activ.TanH()(input_value)

    x = input_value.value
    assert isclose(output.value, (exp(x) - exp(-x)) / (exp(x) + exp(-x))).all()

    # Test Backward pass
    output.backward()
    assert (input_value.grad == 1 - output.value**2).all()


# ====================================================================================================
# Test ReLU activation function


def test_relu(input_value):
    # Test Forward pass
    output = activ.ReLU()(input_value)

    x = input_value.value
    assert (output == maximum(0, x)).all()

    # Test Backward pass
    output.backward()
    assert (input_value.grad[input_value.grad > 0] == 1).all()
    assert (input_value.grad[input_value.grad <= 0] == 0).all()


# ====================================================================================================
# Test LeakyReLU activation function


def test_leaky_relu(input_value):
    # Test Forward pass
    eps = 0.1
    output = activ.LeakyReLU(eps=eps)(input_value)

    x = input_value.value
    assert (output == maximum(eps * x, x)).all()

    # Test Backward pass
    output.backward()
    assert (input_value.grad[input_value.grad > 0] == 1).all()
    assert (input_value.grad[input_value.grad <= 0] == eps).all()


# ====================================================================================================
# Test Swish activation function


def test_swish(input_value):
    # Test Forward pass
    beta = 1
    output = activ.Swish(beta=beta)(input_value)

    x = input_value.value
    sigma = activ.Sigmoid()(beta * x).value
    assert (output == x * sigma).all()

    # Test Backward pass
    output.backward()
    assert (input_value.grad == output.value + sigma *
            (1 - output.value)).all()


# ====================================================================================================
# Test Softmax activation function


def test_softmax(input_value):
    # Test Forward pass
    output = activ.Softmax()(input_value)

    exps = exp(input_value.value)
    sums = sum(exps, axis=0, keepdims=True)

    assert (output == exps / sums).all()

    # Test Backward pass
    output.backward()

    k, n = output.shape
    Sj_matrix = repeat(output.value, k, axis=1)
    Si_matrix = hstack(
        [Sj_matrix[:, (i - k):i].T for i in range(k, (k * n) + 1, k)])
    Sj_diag = hstack([diag(output.value[:, i]) for i in range(n)])

    assert (input_value.grad == Sj_diag - Si_matrix * Sj_matrix).all()


# ====================================================================================================
# Fixtures


@pytest.fixture
def input_value():
    return Tensor([[0.42, 0.32, 0.34], [0.6, 0.1, 1.1]],
                  diff=True,
                  name='test_input')


# ====================================================================================================
