from itertools import count


class _Node:
    ''' A Node in the computation graph

    Can be either a Function or a Tensor.

    Parameters:
    -----------
     - children : varargs, the children of the node.
     - name : string, representation of the node.

    '''

    _id_generator = count()

    def __init__(self, *children: ..., name='Node'):
        # Parse all children that are not Nodes to Tensors
        self.children = []
        for child in children:
            self.add_child(child)

        self.name = name
        self.id: int = next(_Node._id_generator)

    def __eq__(self, other):
        return self.id == other.id

    def __repr__(self):
        return f'<{self.name}>'

    def add_child(self, child) -> None:
        from nujo.autodiff.tensor import Tensor

        self.children.append(child if isinstance(child, _Node) else Tensor(
            child, name=str(child)))
