import pytest
from numpy import ndarray

from nujo import Tensor
from nujo.autodiff.functions import Addition


def test_tensor_value(get_tensors):
    A, B, C = get_tensors

    assert isinstance(A.value, ndarray)
    assert isinstance(B.value, ndarray)
    assert isinstance(C.value, ndarray)


def test_tensor_creator(get_tensors):
    A, B, C = get_tensors

    assert A.creator is None
    assert B.creator is None
    assert isinstance(C.creator, Addition)


def test_tensor_backward(get_tensors):
    A, B, C = get_tensors

    C.backward()

    assert len(C._grad_dependencies) == 0
    assert C.grad == 1

    assert len(A._grad_dependencies) == 1
    assert A._grad_dependencies[0][0] == C
    assert A._grad_dependencies[0][1] == 1
    assert A.grad == 1

    assert len(B._grad_dependencies) == 1
    assert B._grad_dependencies[0][0] == C
    assert B._grad_dependencies[0][1] == 1
    assert B.grad == 1


def test_tensor_transpose(get_tensors):
    A, _, _ = get_tensors

    assert (A.T.value == A.value.T).all()


def test_tensor_shape(get_tensors):
    A, _, _ = get_tensors

    assert A.shape == A.value.shape


def test_tensor_zero_grad(get_tensors):
    A, _, _ = get_tensors

    A.zero_grad()
    assert len(A._grad_dependencies) == 0
    assert A._grad is None
    assert A._T is None


@pytest.fixture
def get_tensors():
    A = Tensor([[1, 2], [3, 4]])
    B = Tensor([[5, 6], [7, 8]])
    C = A + B

    return A, B, C
