import pytest
from numpy import ndarray

from nujo import Tensor
from nujo.autodiff._functions._elementary import _Addition


def test_tensor_value(get_tensors):
    A, B, C = get_tensors

    assert isinstance(A.value, ndarray)
    assert isinstance(B.value, ndarray)
    assert isinstance(C.value, ndarray)


def test_tensor_creator(get_tensors):
    A, B, C = get_tensors

    assert A.creator is None
    assert B.creator is None
    assert isinstance(C.creator, _Addition)


def test_tensor_backward(get_tensors):
    A, B, C = get_tensors

    C.backward()

    assert len(C._grad_dependencies) == 0
    assert (C.grad == 1).all()

    assert len(A._grad_dependencies) == 1
    assert (A._grad_dependencies[0][0] == C).all()
    assert (A._grad_dependencies[0][1] == 1).all()
    assert (A.grad == 1).all()

    assert len(B._grad_dependencies) == 1
    assert (B._grad_dependencies[0][0] == C).all()
    assert (B._grad_dependencies[0][1] == 1).all()
    assert (B.grad == 1).all()


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


def test_tensor_inplace_assignment(get_tensors):
    A, _, C = get_tensors

    A <<= C
    assert A.id != C.id

    assert A.children == C.children or A.children is None
    assert A.creator == C.creator or A.creator is None
    assert (A.value == C.value).all()


@pytest.fixture
def get_tensors():
    A = Tensor([[1, 2], [3, 4]], diff=True)
    B = Tensor([[5, 6], [7, 8]], diff=True)
    C = A + B

    return A, B, C
