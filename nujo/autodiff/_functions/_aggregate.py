from numbers import Number
from typing import List, Optional, Tuple, Union

from numpy import ndarray, ones, prod, sum

from nujo.autodiff.function import Function
from nujo.autodiff.tensor import Tensor

__all__ = [
    '_InnerSum',
    '_InnerProd',
]

# ====================================================================================================


class _InnerSum(Function):
    def __init__(self,
                 input: Union[Tensor, ndarray, List[Number], Number],
                 dim: Optional[int] = None,
                 keepdim=False,
                 name='InnerSum'):

        super(_InnerSum, self).__init__(input, name=name)
        self.dim = dim
        self.keepdim = keepdim

    def forward(self) -> ndarray:
        return sum(self.children[0].value,
                   axis=self.dim,
                   keepdims=self.keepdim)

    def backward(self) -> Tuple[ndarray]:
        return ones(self.children[0].shape),


# ====================================================================================================


class _InnerProd(Function):
    def __init__(self,
                 input: Union[Tensor, ndarray, List[Number], Number],
                 dim: Optional[int] = None,
                 keepdim=False,
                 name='InnerProd'):

        super(_InnerProd, self).__init__(input, name=name)
        self.dim = dim
        self.keepdim = keepdim

        self._output: ndarray = None  # Used to compute the derivative

    def forward(self) -> ndarray:
        self._output = prod(self.children[0].value,
                            axis=self.dim,
                            keepdims=self.keepdim)

        return self._output

    def backward(self) -> Tuple[ndarray]:
        return self._output / self.children[0].value,


# ====================================================================================================
