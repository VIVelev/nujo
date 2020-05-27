from nujo.autodiff._node import _Node


def test_node_equality():
    A, B = _Node(), _Node()

    assert A == A
    assert A != B


def test_node_children():
    A = _Node(_Node(1), _Node(2), 3)

    assert len(A.children) == 3

    assert isinstance(A.children[0], _Node)
    assert A.children[1].children[0] == 2
    assert A.children[2] == 3
