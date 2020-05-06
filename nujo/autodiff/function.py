from abc import abstractmethod
from numbers import Number
from typing import List, Tuple, Union

from numpy import ndarray

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
     - children : varargs, the inpute tensors
     - name : string, the name of the function

    '''
    def __init__(self,
                 *children: Union[Tensor, ndarray, List[Number], Number],
                 name='Function'):

        super(Function, self).__init__(*children, name=name)
        self._reuse = True
        self._z_placeholder: 'Tensor' = None

    def __repr__(self):
        return super(Function, self).__repr__() + f'#{self.id}'

    def _generate_tensor_name(self) -> str:
        return 'Z' + self.__repr__()

    @abstractmethod
    def forward(self) -> ndarray:
        pass

    @abstractmethod
    def backward(self) -> Tuple[ndarray, ...]:
        pass

    def __call__(self) -> Tensor:
        ''' This method controls what gets registered in the
        computation graph and what gets a gradient.

        It also builds the backpropagation graph.

        '''

        z = self.forward()
        if self._z_placeholder is None:
            # Initialize the placeholder
            self._z_placeholder = Tensor(
                z,
                diff=[x.diff for x in self.children],
                creator=self if modes.DIFF_ENABLED else None,
                name=self._generate_tensor_name())

            self._reuse = False

        else:
            self._z_placeholder.value = z
            self._reuse = True

        if modes.DIFF_ENABLED and self._z_placeholder.diff:
            # Compute gradient for this tensor
            for tensor, derivative in zip(self.children, self.backward()):
                if self._reuse:
                    idx = next(i for i, v in enumerate(tensor.backward_depend)
                               if v[0] is self._z_placeholder)

                    tensor.backward_depend[idx][0].value = z
                    tensor.backward_depend[idx][1] = derivative
                else:
                    tensor.add_backward_dep(self._z_placeholder, derivative)

        return self._z_placeholder
