from copy import deepcopy
from math import e

from numpy import around as np_round
from numpy import ceil as np_ceil
from numpy import floor as np_floor

from nujo.autodiff._functions._elementary import _Logarithm, _Power
from nujo.autodiff.tensor import Tensor

__all__ = [
    'log',
    'log2',
    'log10',
    'exp',
    'sqrt',
    'abs',
    'round',
    'ceil',
    'floor',
]

# ====================================================================================================


def log(x: Tensor, base: float = e) -> Tensor:
    return _Logarithm(x, base)()


def log2(x: Tensor) -> Tensor:
    return _Logarithm(x, 2, name='Log2')()


def log10(x: Tensor) -> Tensor:
    return _Logarithm(x, 10, name='Log10')()


# ====================================================================================================


def exp(x: Tensor) -> Tensor:
    return _Power(e, x, name='Exp')()


def sqrt(x: Tensor) -> Tensor:
    return _Power(x, 1 / 2, name='Sqrt')()


def abs(x: Tensor) -> Tensor:
    return sqrt(x**2)


# ====================================================================================================


def round(x: Tensor, inplace=False) -> Tensor:
    rounded = x if inplace else deepcopy(x)
    rounded.name += ' (rounded)'
    rounded.value = np_round(x.value)

    return rounded


def ceil(x: Tensor, inplace=False) -> Tensor:
    ceiled = x if inplace else deepcopy(x)
    ceiled.name += ' (ceiled)'
    ceiled.value = np_ceil(x.value)

    return ceiled


def floor(x: Tensor, inplace=False) -> Tensor:
    floored = x if inplace else deepcopy(x)
    floored.name += ' (floored)'
    floored.value = np_floor(x.value)

    return floored


# ====================================================================================================
