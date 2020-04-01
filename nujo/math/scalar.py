from copy import deepcopy
from math import e
from numbers import Number

from numpy import around as np_round
from numpy import ceil as np_ceil
from numpy import floor as np_floor
from numpy import where

from nujo.autodiff._functions import _Logarithm, _Power
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


def log(x: Number or Tensor, base: float = e) -> Tensor:
    return _Logarithm(x, base)()


def log2(x: Number or Tensor) -> Tensor:
    return _Logarithm(x, 2, name='Log2')()


def log10(x: Number or Tensor) -> Tensor:
    return _Logarithm(x, 10, name='Log10')()


# ====================================================================================================


def exp(x: Number or Tensor) -> Tensor:
    return _Power(e, x, name='Exp')()


def sqrt(x: Number or Tensor) -> Tensor:
    return _Power(x, 1 / 2, name='Sqrt')()


def abs(x: Number or Tensor) -> Tensor:
    return x * where(x < 0, -1, 1)


# ====================================================================================================


def round(x: Number or Tensor) -> Tensor:
    if not isinstance(x, Tensor):
        x = Tensor(x)

    rounded = deepcopy(x)
    rounded.name += ' (rounded)'
    rounded.value = np_round(x.value)

    return rounded


def ceil(x: Number or Tensor) -> Tensor:
    if not isinstance(x, Tensor):
        x = Tensor(x)

    ceiled = deepcopy(x)
    ceiled.name += ' (ceiled)'
    ceiled.value = np_ceil(x.value)

    return ceiled


def floor(x: Number or Tensor) -> Tensor:
    if not isinstance(x, Tensor):
        x = Tensor(x)

    floored = deepcopy(x)
    floored.name += ' (floored)'
    floored.value = np_floor(x.value)

    return floored


# ====================================================================================================
