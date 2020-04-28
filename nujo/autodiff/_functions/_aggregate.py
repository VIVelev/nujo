from numpy import eye, ndarray, ones, prod, sum

from nujo._typing import Union, _numerical
from nujo.autodiff.function import Function
from nujo.autodiff.tensor import Tensor

__all__ = [
    '_InnerSum',
    '_InnerProd',
]

# ====================================================================================================


class _InnerSum(Function):
    def __init__(self,
                 input: Union[Tensor, _numerical],
                 dim: int = None,
                 keepdim=False,
                 name='InnerSum'):
        super(_InnerSum, self).__init__(input, name=name)
        self.dim = dim
        self.keepdim = keepdim

    def forward(self) -> ndarray:
        return sum(self.children[0].value,
                   axis=self.dim,
                   keepdims=self.keepdim)

    def backward(self) -> tuple:
        return ones(self.children[0].shape),


# ====================================================================================================


class _InnerProd(Function):
    def __init__(self,
                 input: Union[Tensor, _numerical],
                 dim: int = None,
                 keepdim=False,
                 name='InnerProd'):
        super(_InnerProd, self).__init__(input, name=name)
        self.dim = dim
        self.keepdim = keepdim

    def forward(self) -> ndarray:
        return prod(self.children[0].value,
                    axis=self.dim,
                    keepdims=self.keepdim)

    def backward(self) -> tuple:
        mask = -(eye(self.children[0].shape[0]) - 1)
        matrix = self.children[0].value.repeat(self.children[0].shape[0],
                                               axis=-1)

        return prod(mask * matrix, axis=0),


# ====================================================================================================
