from math import e
from numbers import Number
from typing import List, Union

from numpy import identity, ndarray

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
        self.i = identity(input.children[0].shape[0])

    def forward(self) -> ndarray:
        return (e**(self.i * self.children[0]) -
                e**-(self.i * self.children[0])) /\
                (2 * self.i)

    def backward(self) -> ndarray:
        # cos(X)
        return _Cos(self.children[0])()


# ====================================================================================================


class _Cos(Function):
    ''' Differable cosine function

    cos(X) = (e^iX + e^-iX) / i

    '''
    def __init__(self, input: Union[Tensor, ndarray, List[Number], Number]):
        super(_Cos, self).__init__(input)
        self.i = identity(input.children[0].shape[0])

    def forward(self) -> ndarray:
        return (e**(self.i * self.children[0]) +
                e**-(self.i * self.children[0])) /\
                self.i

    def backward(self) -> ndarray:
        # sin(X)
        return _Sin(self.children[0])()


# ====================================================================================================


class _Tan(Function):
    ''' Differable tangent function

    tan(X) = sin(X) / cos(X)

    '''
    def __init__(self, input: Union[Tensor, ndarray, List[Number], Number]):
        super(_Tan, self).__init__(input)

    def forward(self) -> ndarray:
        return _Sin(self.children[0])() /\
               _Cos(self.children[0])()

    def backward(self) -> ndarray:
        # sec^2(X)
        # sec(X) = 1 / cos(X)
        return (1 / _Cos(self.children[0])())**2
