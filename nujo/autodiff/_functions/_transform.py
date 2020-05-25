from numbers import Number
from typing import List, Optional, Tuple, Union

from numpy import ndarray

from nujo.autodiff.function import Function
from nujo.autodiff.tensor import Tensor

__all__ = [
    '_Reshape',
    '_Transpose',
]

# ====================================================================================================


class _Reshape(Function):
    def __init__(self, input: Union[Tensor, ndarray, List[Number], Number],
                 shape: Tuple[int, ...]):

        super(_Reshape, self).__init__(input, name=self.__class__.__name__)
        self.shape = shape
        self._input_shape = self.children[0].shape

    def forward(self) -> ndarray:
        return self.children[0].value.reshape(self.shape)

    def backward(self) -> Tuple[ndarray]:
        return self._input_shape,


# ====================================================================================================


class _Transpose(Function):
    def __init__(self,
                 input: Union[Tensor, ndarray, List[Number], Number],
                 dims: Optional[Tuple[int, ...]] = None):

        super(_Transpose, self).__init__(input, name=self.__class__.__name__)
        self.dims = dims if dims is not None else reversed(
            range(len(self.dims)))
        self._detranspose_dims = sorted(range(len(self.dims)),
                                        key=lambda idx: self.dims[idx])

    def forward(self) -> ndarray:
        return self.children[0].value.transpose(*self.dims)

    def backward(self) -> Tuple[ndarray]:
        return self._detranspose_dims,


# ====================================================================================================
