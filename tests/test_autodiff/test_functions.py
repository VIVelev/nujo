from numbers import Number

import pytest
from numpy import allclose, log, log2, ndarray

import nujo.autodiff._functions._elementary as funcs
from nujo import Tensor

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
    grad = add.backward()
    assert len(grad) == 2

    assert isinstance(grad[0], Number) or isinstance(grad[0], ndarray)
    assert isinstance(grad[1], Number) or isinstance(grad[1], ndarray)

    # Test Derivative computation
    assert (grad[0] == 1).all()
    assert (grad[1] == 1).all()


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
    grad = neg.backward()
    assert len(grad) == 1

    assert isinstance(grad[0], Number) or isinstance(grad[0], ndarray)

    # Test Derivative computation
    assert (grad[0] == -1).all()


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
    grad = mul.backward()
    assert len(grad) == 2

    assert isinstance(grad[0], Number) or isinstance(grad[0], ndarray)
    assert isinstance(grad[1], Number) or isinstance(grad[1], ndarray)

    # Test Derivative computation
    assert (grad[0] == B.value).all()
    assert (grad[1] == A.value).all()


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
    grad = recipr.backward()
    assert len(grad) == 1

    assert isinstance(grad[0], Number) or isinstance(grad[0], ndarray)

    # Test Derivative computation
    assert (grad[0] == -1 / (A.value**2)).all()


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
    grad = pow.backward()
    assert len(grad) == 2

    assert isinstance(grad[0], Number) or isinstance(grad[0], ndarray)
    assert isinstance(grad[1], Number) or isinstance(grad[1], ndarray)

    # Test Derivative computation
    assert (grad[0] == 2 * A.value).all()
    assert grad[1] == 1


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
    grad = log_2.backward()
    assert len(grad) == 2

    assert isinstance(grad[0], Number) or isinstance(grad[0], ndarray)
    assert isinstance(grad[1], Number) or isinstance(grad[1], ndarray)

    # Test Derivative computation
    assert allclose(grad[0], 1 / (A.value * log(2)))
    assert grad[1] == 1


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
    grad = matmul.backward()
    assert len(grad) == 2

    assert isinstance(grad[0], Number) or isinstance(grad[0], ndarray)
    assert isinstance(grad[1], Number) or isinstance(grad[1], ndarray)

    # Test Derivative computation
    assert (grad[0] == B).all()
    assert (grad[1] == A).all()


# ====================================================================================================
# Unit Test fixtures


@pytest.fixture
def inputs():
    A = Tensor([[1, 2], [3, 4]])
    B = Tensor([[5, 6], [7, 8]])

    return A, B


# ====================================================================================================
