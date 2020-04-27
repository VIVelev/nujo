from copy import deepcopy
from math import e

from numpy import around as np_round
from numpy import ceil as np_ceil
from numpy import floor as np_floor
from numpy import where

from nujo._typing import Union, _numerical
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


def log(x: Union[Tensor, _numerical], base: float = e) -> Tensor:
    return _Logarithm(x, base)()


def log2(x: Union[Tensor, _numerical]) -> Tensor:
    return _Logarithm(x, 2, name='Log2')()


def log10(x: Union[Tensor, _numerical]) -> Tensor:
    return _Logarithm(x, 10, name='Log10')()


# ====================================================================================================


def exp(x: Union[Tensor, _numerical]) -> Tensor:
    return _Power(e, x, name='Exp')()


def sqrt(x: Union[Tensor, _numerical]) -> Tensor:
    return _Power(x, 1 / 2, name='Sqrt')()


def abs(x: Union[Tensor, _numerical]) -> Tensor:
    return x * where(x < 0, -1, 1)


# ====================================================================================================


def round(x: Union[Tensor, _numerical], inplace=False) -> Tensor:
    if not isinstance(x, Tensor):
        x = Tensor(x)

    rounded = x if inplace else deepcopy(x)
    rounded.name += ' (rounded)'
    rounded.value = np_round(x.value)

    return rounded


def ceil(x: Union[Tensor, _numerical], inplace=False) -> Tensor:
    if not isinstance(x, Tensor):
        x = Tensor(x)

    ceiled = x if inplace else deepcopy(x)
    ceiled.name += ' (ceiled)'
    ceiled.value = np_ceil(x.value)

    return ceiled


def floor(x: Union[Tensor, _numerical], inplace=False) -> Tensor:
    if not isinstance(x, Tensor):
        x = Tensor(x)

    floored = x if inplace else deepcopy(x)
    floored.name += ' (floored)'
    floored.value = np_floor(x.value)

    return floored


# ====================================================================================================
