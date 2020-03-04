import pytest

from nujo.autodiff.node import Node


def test_node(get_nodes):
    A, B = get_nodes

    assert A == A
    assert A != B

    A.add_child(1)
    A.add_child(2)
    A.add_child(3)

    assert len(A.children) == 3
    assert isinstance(A.children[-1], Node)
    assert A.children[-1].value == 3


@pytest.fixture
def get_nodes():
    return Node(), Node()
