from numbers import Number
from typing import List, Union

from numpy import cos, ndarray, sin, tan

from nujo.autodiff.function import Function
from nujo.autodiff.tensor import Tensor

__all__ = ['_Sin', '_Cos', '_Tan']

# ====================================================================================================


class _Sin(Function):
    ''' Differable sine function

    sin(X) = (e^iX - e^-iX) / 2i

    '''
    def __init__(self, input: Union[Tensor, ndarray, List[Number], Number]):
        super(_Sin, self).__init__(input)

    def forward(self) -> ndarray:
        return sin(self.children[0].value)

    def backward(self) -> Function.T:
        # cos(X)
        return Tensor(_Cos(self.children[0].value)())


# ====================================================================================================


class _Cos(Function):
    ''' Differable cosine function

    cos(X) = (e^iX + e^-iX) / i

    '''
    def __init__(self, input: Union[Tensor, ndarray, List[Number], Number]):
        super(_Cos, self).__init__(input)

    def forward(self) -> ndarray:
        return cos(self.children[0].value)

    def backward(self) -> Function.T:
        # -sin(X)
        return Tensor(-(_Sin(self.children[0].value)()))


# ====================================================================================================


class _Tan(Function):
    ''' Differable tangent function

    tan(X) = sin(X) / cos(X)

    '''
    def __init__(self, input: Union[Tensor, ndarray, List[Number], Number]):
        super(_Tan, self).__init__(input)

    def forward(self) -> ndarray:
        return tan(self.children[0].value)

    def backward(self) -> Function.T:
        # sec^2(X)
        return Tensor((1 / _Cos(self.children[0].value)())**2)


# ====================================================================================================
