from nujo.autodiff.utils import counter


class Node:
    ''' A Node in the computation graph

    Can be either a Function or a Tensor.

    Parameters:
    -----------
    children : varargs, the children of the node.
    name : string, representation of the node.

    '''

    epsilon = 1e-18
    id_generator = counter()

    def __init__(self, *children, name='<Node>'):
        self.children = []
        for child in children:
            self.add_child(child)

        self.name = name
        self.id = Node.id_generator.get()

    def __eq__(self, other):
        return self.id == other.id

    def add_child(self, child) -> None:
        from nujo.autodiff.tensor import Tensor

        self.children.append(
            child if isinstance(child, Node) else Tensor(child))
