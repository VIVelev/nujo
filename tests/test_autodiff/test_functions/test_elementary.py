import pytest
from numpy import allclose, log, log2

import nujo.autodiff._functions._elementary as funcs
from nujo import Tensor, ones

# ====================================================================================================
# Unit Testing Addition


def test_addition(inputs):
    A, B = inputs
    add = funcs._Addition(A, B)

    # Test Forwardprop
    C = add()
    assert isinstance(C, Tensor)
    assert (A.value + B.value == C.value).all()

    # Test Backprop
    grad_A, grad_B = add.backward(0, Tensor(1)), add.backward(1, Tensor(1))

    assert isinstance(grad_A, Tensor)
    assert isinstance(grad_B, Tensor)

    # Test Derivative computation
    assert grad_A.shape == A.shape
    assert (grad_A == 1).all()

    assert grad_B.shape == B.shape
    assert (grad_B == 1).all()


# ====================================================================================================
# Unit Testing Negation


def test_negation(inputs):
    A, _ = inputs
    neg = funcs._Negation(A)

    # Test Forwardprop
    C = neg()
    assert isinstance(C, Tensor)
    assert (-A.value == C.value).all()

    # Test Backprop
    grad_A = neg.backward(0, Tensor(1))

    assert isinstance(grad_A, Tensor)

    # Test Derivative computation
    assert grad_A.shape == A.shape
    assert (grad_A == -1).all()


# ====================================================================================================
# Unit Testing Multiplication


def test_multiplication(inputs):
    A, B = inputs
    mul = funcs._Multiplication(A, B)

    # Test Forwardprop
    C = mul()
    assert isinstance(C, Tensor)
    assert (A.value * B.value == C.value).all()

    # Test Backprop
    grad_A, grad_B = mul.backward(0, Tensor(1)), mul.backward(1, Tensor(1))

    assert isinstance(grad_A, Tensor)
    assert isinstance(grad_B, Tensor)

    # Test Derivative computation
    assert grad_A.shape == A.shape
    assert (grad_A == B).all()

    assert grad_B.shape == B.shape
    assert (grad_B == A).all()


# ====================================================================================================
# Unit Testing Reciprocal


def test_reciprocal(inputs):
    A, _ = inputs
    recipr = funcs._Reciprocal(A)

    # Test Forwardprop
    C = recipr()
    assert isinstance(C, Tensor)
    assert (1 / A.value == C.value).all()

    # Test Backprop
    grad_A = recipr.backward(0, Tensor(1))

    assert isinstance(grad_A, Tensor)

    # Test Derivative computation
    assert grad_A.shape == A.shape
    assert (grad_A == -1 / A**2).all()


# ====================================================================================================
# Unit Testing Power


def test_power(inputs):
    A, _ = inputs
    pow = funcs._Power(A, 2)

    # Test Forwardprop
    C = pow()
    assert isinstance(C, Tensor)
    assert (A.value**2 == C.value).all()

    # Test Backprop
    grad_A, grad_B = pow.backward(0, Tensor(1)), pow.backward(1, Tensor(1))

    assert isinstance(grad_A, Tensor)
    assert isinstance(grad_B, Tensor)

    # Test Derivative computation
    assert grad_A.shape == A.shape
    assert (grad_A == 2 * A).all()

    assert grad_B == 1


# ====================================================================================================
# Unit Testing Logarithm


def test_logarithm(inputs):
    A, _ = inputs
    log_2 = funcs._Logarithm(A, 2)  # log_2(A)

    # Test Forwardprop
    C = log_2()
    assert isinstance(C, Tensor)
    assert allclose(log2(A.value), C.value)

    # Test Backprop
    grad_A, grad_B = log_2.backward(0, Tensor(1)), log_2.backward(1, Tensor(1))

    assert isinstance(grad_A, Tensor)
    assert isinstance(grad_B, Tensor)

    # Test Derivative computation
    assert grad_A.shape == A.shape
    assert allclose(grad_A.value, 1 / (A.value * log(2)))

    assert grad_B == 1


# ====================================================================================================
# Unit Testing Matrix Multiplication


def test_matrixmul(inputs):
    A, B = inputs
    matmul = funcs._MatrixMul(A, B)

    # Test Forwardprop
    C = matmul()
    assert isinstance(C, Tensor)
    assert (A.value @ B.value == C.value).all()

    # Test Backprop
    output_shape = (A.shape[0], B.shape[1])
    doutput = ones(*output_shape)
    grad_A, grad_B = matmul.backward(0, doutput), matmul.backward(1, doutput)

    assert isinstance(grad_A, Tensor)
    assert isinstance(grad_B, Tensor)

    # Test Derivative computation
    assert grad_A.shape[0] == A.shape[1]
    assert (grad_A == doutput @ B.T).all()

    assert grad_B.shape[1] == B.shape[0]
    assert (grad_B == (doutput.T @ A).T).all()


# ====================================================================================================
# Unit Test fixtures


@pytest.fixture
def inputs():
    A = Tensor([[1, 2], [3, 4]])
    B = Tensor([[5, 6], [7, 8]])

    return A, B


# ====================================================================================================
