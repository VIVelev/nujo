from copy import deepcopy
from math import e
from numbers import Number

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


def log(x: Tensor or Number, base: float = e) -> Tensor:
    return Logarithm(x, base)()


def log2(x: Tensor or Number) -> Tensor:
    return Logarithm(x, 2, name='<Log2>')()


def log10(x: Tensor or Number) -> Tensor:
    return Logarithm(x, 10, name='<Log10>')()


# ====================================================================================================


def exp(x: Tensor or Number) -> Tensor:
    return Power(e, x, name='<Exp>')()


def sqrt(x: Tensor or Number) -> Tensor:
    return Power(x, 1 / 2, name='<Sqrt>')()


def abs(x: Tensor or Number) -> Tensor:
    func = sqrt(x**2)
    func.name = '<Abs>'
    return func


# ====================================================================================================


def round(x: Tensor or Number) -> Tensor:
    if not isinstance(x, Tensor):
        x = Tensor(x)

    rounded = deepcopy(x)
    rounded.name += ' (rounded)'
    rounded.value = np_round(x.value)

    return rounded


def ceil(x: Tensor or Number) -> Tensor:
    if not isinstance(x, Tensor):
        x = Tensor(x)

    ceiled = deepcopy(x)
    ceiled.name += ' (ceiled)'
    ceiled.value = np_ceil(x.value)

    return ceiled


def floor(x: Tensor or Number) -> Tensor:
    if not isinstance(x, Tensor):
        x = Tensor(x)

    floored = deepcopy(x)
    floored.name += ' (floored)'
    floored.value = np_floor(x.value)

    return floored


# ====================================================================================================
