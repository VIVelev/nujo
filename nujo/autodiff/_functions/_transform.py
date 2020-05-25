from numbers import Number
from typing import List, Tuple, Union

from numpy import ndarray, prod, reshape, transpose

from nujo.autodiff.function import Function
from nujo.autodiff.tensor import Tensor

__all__ = [
    '_Reshape',
]


class _Reshape(Function):
    def __init__(self,
                 input: Union[Tensor, ndarray, List[Number], Number],
                 shape: Tuple[int, ...],
                 name='Reshape'):

        super(_Reshape, self).__init__(input, name=name)
        self.shape = shape
        self._input_shape = self.children[0].shape

    def forward(self) -> ndarray:
        return reshape(self.children[0].value, self.shape)

    def backward(self) -> Tuple[ndarray]:
        return self._input_shape,
