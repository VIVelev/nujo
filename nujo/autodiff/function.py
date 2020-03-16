from abc import abstractmethod

from numpy import array, ndarray

from nujo.autodiff.modes import DIFF_ENABLED
from nujo.autodiff.node import Node
from nujo.autodiff.tensor import Tensor


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
    def __init__(self, *children: Tensor, name='<Function>') -> None:
        super(Function, self).__init__(*children, name=name)

    def __repr__(self) -> str:
        return self.name + f'#{self.id}'

    def _generate_tensor_name(self) -> str:
        return 'Z' + self.__repr__()

    @abstractmethod
    def forward(self) -> ndarray:
        pass

    @abstractmethod
    def backward(self) -> tuple:
        pass

    def __call__(self) -> Tensor:
        z = self.forward()
        if not isinstance(z, Tensor):
            z = Tensor(z, creator=self, name=self._generate_tensor_name())

        if DIFF_ENABLED:
            for tensor, derivative in zip(self.children, self.backward()):
                tensor.add_grad_dependency(z, array(derivative))

        return z
