from itertools import count
from typing import Any


class _Node:
    ''' A Node in the computation graph

    Can be either a Function or a Tensor.

    Parameters:
    -----------
     - children : varargs, the children of the node
     - name : string, representation of the node

    '''

    _id_generator = count()

    def __init__(self, *children: Any, name='Node'):
        self.children = list(children)
        self.name = name
        self.id: int = next(_Node._id_generator)

    def __eq__(self, other):
        return self.id == other.id

    def __repr__(self):
        return f'<{self.name}>'
