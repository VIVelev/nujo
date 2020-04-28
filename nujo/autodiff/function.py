from abc import abstractmethod

from numpy import ndarray

from nujo._typing import Union, _numerical
from nujo.autodiff import modes
from nujo.autodiff._node import _Node
from nujo.autodiff.tensor import Tensor


class Function(_Node):
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
    def __init__(self,
                 *children: Union[Tensor, _numerical],
                 name='Function') -> None:
        super(Function, self).__init__(*children, name=name)

    def __repr__(self):
        return super(Function, self).__repr__() + f'#{self.id}'

    def _generate_tensor_name(self) -> str:
        return 'Z' + self.__repr__()

    @abstractmethod
    def forward(self) -> ndarray:
        pass

    @abstractmethod
    def backward(self) -> tuple:
        pass

    def __call__(self) -> Tensor:
        ''' This method controls what gets registered in the
        computation graph and what gets a gradient.
        '''

        z = self.forward()
        if not isinstance(z, Tensor):
            # Register in the computation graph only if in diff mode
            z = Tensor(z,
                       diff=any([x.diff for x in self.children]),
                       creator=self if modes.DIFF_ENABLED else None,
                       name=self._generate_tensor_name())

        if modes.DIFF_ENABLED and z.diff:
            # Compute gradient for this tensor
            for tensor, derivative in zip(self.children, self.backward()):
                tensor.add_grad_dependency(
                    z, Tensor(derivative, name=f'weight[{z.name}]'))

        return z
