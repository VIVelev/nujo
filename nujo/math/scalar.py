from copy import deepcopy
from math import e

from numpy import around as np_round
from numpy import ceil as np_ceil
from numpy import floor as np_floor

from nujo.autodiff import Tensor
from nujo.autodiff.functions import Logarithm, Power

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
    return Logarithm(x, base)()


def log2(x: Tensor) -> Tensor:
    return Logarithm(x, 2, name='<Log2>')()


def log10(x: Tensor) -> Tensor:
    return Logarithm(x, 10, name='<Log10>')()


# ====================================================================================================


def exp(x: Tensor) -> Tensor:
    return Power(e, x, name='<Exp>')()


def sqrt(x: Tensor) -> Tensor:
    return Power(x, 1 / 2, name='<Sqrt>')()


def abs(x: Tensor) -> Tensor:
    return sqrt(x**2)


# ====================================================================================================


def round(x: Tensor) -> Tensor:
    rounded = deepcopy(x)
    rounded.name += ' (rounded)'
    rounded.value = np_round(x.value)

    return rounded


def ceil(x: Tensor) -> Tensor:
    ceiled = deepcopy(x)
    ceiled.name += ' (ceiled)'
    ceiled.value = np_ceil(x.value)

    return ceiled


def floor(x: Tensor) -> Tensor:
    floored = deepcopy(x)
    floored.name += ' (floored)'
    floored.value = np_floor(x.value)

    return floored


# ====================================================================================================
