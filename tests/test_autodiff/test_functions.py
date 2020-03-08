import pytest

import nujo.autodiff.functions as funcs
from nujo import Tensor


def test_addition(get_tensors):
    A, B = get_tensors
    add = funcs.Addition(A, B)
    C = add()

    assert (A.value + B.value).all() == C.value.all()
    assert len(add.backward()) == 2


def test_negation(get_tensors):
    A, _ = get_tensors
    neg = funcs.Negation(A)
    C = neg()

    assert (-A.value).all() == C.value.all()
    assert len(neg.backward()) == 1


def test_multiplication(get_tensors):
    A, B = get_tensors
    mul = funcs.Multiplication(A, B)
    C = mul()

    assert (A.value * B.value).all() == C.value.all()
    assert len(mul.backward()) == 2


def test_reciprocal(get_tensors):
    A, _ = get_tensors
    recipr = funcs.Reciprocal(A)
    C = recipr()

    assert (1 / A.value).all() == C.value.all()
    assert len(recipr.backward()) == 1


def test_power(get_tensors):
    A, _ = get_tensors
    pow = funcs.Power(A, 2)
    C = pow()

    assert (A.value**2).all() == C.value.all()
    assert len(pow.backward()) == 2


def test_MatrixMul(get_tensors):
    A, B = get_tensors
    matmul = funcs.MatrixMul(A, B)
    C = matmul()

    assert (A.value @ B.value).all() == C.value.all()
    assert len(matmul.backward()) == len(get_tensors)


@pytest.fixture
def get_tensors():
    A = Tensor([[1, 2], [3, 4]])
    B = Tensor([[5, 6], [7, 8]])

    return A, B
