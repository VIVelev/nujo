from numbers import Number

import pytest
from numpy import ndarray

import nujo.autodiff.functions as funcs
from nujo import Tensor

# ====================================================================================================
# Unit Testing Addition


def test_addition(get_tensors):
    A, B = get_tensors
    add = funcs.Addition(A, B)

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
    assert grad[0] == 1
    assert grad[1] == 1


# ====================================================================================================
# Unit Testing Negation


def test_negation(get_tensors):
    A, _ = get_tensors
    neg = funcs.Negation(A)

    # Test Forwardprop
    C = neg()
    assert isinstance(C, Tensor)
    assert (-A.value == C.value).all()

    # Test Backprop
    grad = neg.backward()
    assert len(grad) == 1

    assert isinstance(grad[0], Number) or isinstance(grad[0], ndarray)

    # Test Derivative computation
    assert grad[0] == -1


# ====================================================================================================
# Unit Testing Multiplication


def test_multiplication(get_tensors):
    A, B = get_tensors
    mul = funcs.Multiplication(A, B)

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


def test_reciprocal(get_tensors):
    A, _ = get_tensors
    recipr = funcs.Reciprocal(A)

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


def test_power(get_tensors):
    A, _ = get_tensors
    pow = funcs.Power(A, 2)

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
# Unit Testing Matrix Multiplication


def test_matrixmul(get_tensors):
    A, B = get_tensors
    matmul = funcs.MatrixMul(A, B)

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
    dA, dB = funcs.MatrixMul.differentiate(A, B)
    assert (grad[0] == dA).all()
    assert (grad[1] == dB).all()


# ====================================================================================================
# Unit Test fixtures


@pytest.fixture
def get_tensors():
    A = Tensor([[1, 2], [3, 4]])
    B = Tensor([[5, 6], [7, 8]])

    return A, B


# ====================================================================================================
