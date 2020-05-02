import pytest

from nujo.autodiff._node import _Node


def test_node_equality(nodes):
    A, B = nodes

    assert A == A
    assert A != B


def test_node_children(nodes):
    A, B = nodes

    A.add_child(1)
    A.add_child(2)
    A.add_child(3)

    assert len(A.children) == 3
    assert isinstance(A.children[-1], _Node)
    assert A.children[-1].value == 3


@pytest.fixture
def nodes():
    return _Node(), _Node()
