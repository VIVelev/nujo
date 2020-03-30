import pytest

from nujo.flow import Flow


def test_chaining(flows):
    _, _, C = flows

    assert repr(C) == '<|A >> B>'
    assert len(C.subflows) == 2


def test_forward(flows):
    A, _, C = flows

    assert A(42) == 42
    assert C(42) == 42


def test_append(flows):
    A, B, _ = flows

    assert not A.is_supflow
    A.append(B)
    assert A.is_supflow

    assert repr(A) == '<|A >> B>'
    assert A(42) == 42


def test_pop(flows):
    A, B, C = flows

    poped = C.pop()
    assert poped is B
    assert C.is_supflow

    assert C.name is A.name
    assert C(42) == A(42) == 42


@pytest.fixture
def flows():
    A = Flow('A')
    B = Flow('B')
    C = A >> B

    return A, B, C
