# from typing import Union

from math import e

from numpy import identity, ndarray

from nujo.autodiff.function import Function
from nujo.autodiff.tensor import Tensor

__all__ = ['_Sin', '_Cos', '_Tan']

# ====================================================================================================


class _Sin(Function):
    ''' Differable sine function

    sin(X) = (e^iX - e^-iX) / 2i

    '''
    def __init__(self, input: Tensor):
        super(_Sin, self).__init__(input)
        self.i = identity(input.children[0].value.shape[0])

    def forward(self) -> ndarray:
        return (e**(self.i * self.children[0]) -
                e**-(self.i * self.children[0])) /\
                (2 * self.i)

    def backward(self):
        pass


# ====================================================================================================


class _Cos(Function):
    ''' Differable cosine function

    cos(X) = (e^iX + e^-iX) / i

    '''
    def __init__(self, input):
        super(_Cos, self).__init__(input)
        self.i = identity(input.children[0].value.shape[0])

    def forward(self) -> ndarray:
        return (e**(self.i * self.children[0]) +
                e**-(self.i * self.children[0])) /\
                self.i


# ====================================================================================================


class _Tan(Function):
    ''' Differable tangent function

    tan(X) = sin(X) / cos(X)

    '''
    def __init__(self, input):
        pass

    def forward(self) -> ndarray:
        return _Sin(self.children[0].value)() /\
               _Cos(self.children[0].value)()

    def backward(self):
        pass
