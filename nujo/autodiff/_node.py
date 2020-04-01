from nujo.autodiff._utils import _counter


class _Node:
    ''' A Node in the computation graph

    Can be either a Function or a Tensor.

    Parameters:
    -----------
    children : varargs, the children of the node.
    name : string, representation of the node.

    '''

    id_generator = _counter()

    def __init__(self, *children, name='Node'):
        self.children = []
        for child in children:
            self.add_child(child)

        self.name = name
        self.id = _Node.id_generator.get()

    def __eq__(self, other):
        return self.id == other.id

    def __repr__(self):
        return f'<{self.name}>'

    def add_child(self, child) -> None:
        from nujo.autodiff.tensor import Tensor

        self.children.append(
            child if isinstance(child, _Node) else Tensor(child))
