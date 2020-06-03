import pytest
from numpy import allclose, array, cos, sin, tan

import nujo.autodiff._functions._trigonometric as funcs
from nujo import Tensor

# ====================================================================================================
# Unit Testing Sine


def test_sin(input):
    A = input
    sine = funcs._Sin(A)

    # Test Forwardprop
    C = sine()
    assert isinstance(C, Tensor)
    assert allclose(sin(A), C.value)

    # Test Backprop
    grad_A = sine.backward(0, 0)
    assert isinstance(grad_A, Tensor)

    # Test Derivative computation
    assert grad_A.shape == A.shape
    assert allclose(grad_A.value, cos(A))


# ====================================================================================================
# Unit Testing Cosine


def test_cos(input):
    A = input
    cosine = funcs._Cos(A)

    # Test Forwardprop
    C = cosine()
    assert isinstance(C, Tensor)
    assert allclose(cos(A), C.value)

    # Test Backprop
    grad_A = cosine.backward(0, 0)
    assert isinstance(grad_A, Tensor)

    # Test Derivative computation
    assert grad_A.shape == A.shape
    assert allclose(grad_A.value, -sin(A))


# ====================================================================================================
# Unit Testing Tangent


def test_tan(input):
    A = input
    tangent = funcs._Tan(A)

    # Test Forwardprop
    C = tangent()
    assert isinstance(C, Tensor)
    assert allclose(tan(A), C.value)

    # Test Backprop
    grad_A = tangent.backward(0, 0)
    assert isinstance(grad_A, Tensor)

    # Test Derivative computation
    assert grad_A.shape == A.shape
    assert allclose(grad_A.value, (1 / cos(A))**2)


# ====================================================================================================
# Unit Test fixtures


@pytest.fixture
def input():
    A = array([[1, 2], [3, 4]])

    return A


# ====================================================================================================
