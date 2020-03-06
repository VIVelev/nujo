from abc import abstractmethod

from nujo.autodiff.modes import DIFF_ENABLED
from nujo.autodiff.node import Node


class Function(Node):
    ''' Base Class for functions

    Functions are applied to tensors.

    A Function takes multiple tensors as input
    and produces only one tensor as output.

    Functions do not change tensors in-place.

    Parameters:
    -----------
    children : varargs, the inpute tensors
    name : string, the name of the function

    '''
    def __init__(self, *children, name='<Function>'):
        super(Function, self).__init__(*children, name)

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def backward(self):
        pass

    def __call__(self):
        z = self.forward()

        if DIFF_ENABLED:
            for tensor, derivative in zip(self.children, self.backward()):
                tensor.add_grad_dependency(z, derivative)

        return z
