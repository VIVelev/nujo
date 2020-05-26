import pytest
from numpy import expand_dims, ndarray

from nujo import Tensor
from nujo.autodiff._functions._elementary import _Addition

# ====================================================================================================
# Test Tensor value and creator properties


def test_tensor_value(tensors):
    A, B, C = tensors

    assert isinstance(A.value, ndarray)
    assert isinstance(B.value, ndarray)
    assert isinstance(C.value, ndarray)


def test_tensor_creator(tensors):
    A, B, C = tensors

    assert A.creator is None
    assert B.creator is None
    assert isinstance(C.creator, _Addition)


# ====================================================================================================
# Test Tensor backward method


def test_tensor_backward(tensors):
    A, B, C = tensors

    C.backward()

    assert len(C.parents_outputs) == len(C.weights) == 0
    assert (C.grad == 1).all()

    assert len(A.parents_outputs) == len(A.weights) == 1
    assert (A.parents_outputs[0] == C).all()
    assert (A.weights[0] == 1).all()
    assert (A.grad == 1).all()

    assert len(B.parents_outputs) == len(B.weights) == 1
    assert (B.parents_outputs[0] == C).all()
    assert (B.weights[0] == 1).all()
    assert (B.grad == 1).all()


# ====================================================================================================
# Test Tensor transpose and shape manipulation
# methods: reshape, repeat, squeeze, unsqueeze


def test_tensor_transpose(tensors):
    A, _, _ = tensors

    assert (A.T.value == A.value.T).all()


def test_tensor_shape_manipulation(tensors):
    A, _, _ = tensors
    assert A.shape == A.value.shape

    A, A_np = A.reshape(-1, 1), A.value.reshape(-1, 1)
    assert (A == A_np).all()

    assert (A.squeeze(1) == A_np.squeeze(1)).all()
    assert (A.unsqueeze(1) == expand_dims(A_np, 1)).all()


# ====================================================================================================
# Test gradient cleaning method


def test_tensor_zero_grad(tensors):
    A, _, _ = tensors

    A.zero_grad()
    assert (A.grad == 0).all()
    assert A._grad_is_zeroed


# ====================================================================================================
# Test inplace assignment operator


def test_tensor_inplace_assignment(tensors):
    A, _, C = tensors

    A <<= C
    assert A.id != C.id

    assert A.children == C.children or A.children is None
    assert A.creator == C.creator or A.creator is None
    assert (A.value == C.value).all()
    assert (A.grad == C.grad).all()
    assert A._grad_is_zeroed == C._grad_is_zeroed


# ====================================================================================================
# Unit Test fixtures


@pytest.fixture
def tensors():
    A = Tensor([[1, 2], [3, 4]], diff=True)
    B = Tensor([[5, 6], [7, 8]], diff=True)
    C = A + B

    return A, B, C


# ====================================================================================================
