from nujo.autodiff.misc import counter


class Node:
    ''' A Node in the computation graph.

    Can be either a function or a tensor.

    Parameters:
    -----------
    children : the children of the node.
    name : string, representation of the node.

    '''

    id_generator = counter()

    def __init__(self, children, name='<Node>'):
        self.children = []
        for child in children:
            self.add_child(child)

        self.name = name
        self.id = Node.id_generator.get()

    def __eq__(self, other):
        return self.name == other.name and self.id == other.id

    def add_child(self, child):
        from nujo.autodiff.tensor import Tensor

        self.children.append(
            child if isinstance(child, Node) else Tensor(child))
