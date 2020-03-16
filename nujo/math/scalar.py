from copy import deepcopy

from numpy import around as np_round
from numpy import ceil as np_ceil
from numpy import floor as np_floor

from nujo.autodiff.functions import Power
from nujo.autodiff.tensor import Tensor

__all__ = [
    'sqrt',
    'abs',
    'round',
    'ceil',
    'floor',
]

# ====================================================================================================


def sqrt(input: Tensor) -> Tensor:
    return Power(input, 1 / 2)()


def abs(input: Tensor) -> Tensor:
    return sqrt(input**2)


def round(input: Tensor) -> Tensor:
    rounded = deepcopy(input)
    rounded.name += ' (rounded)'
    rounded.value = np_round(input.value)

    return rounded


def ceil(input: Tensor) -> Tensor:
    ceiled = deepcopy(input)
    ceiled.name += ' (ceiled)'
    ceiled.value = np_ceil(input.value)

    return ceiled


def floor(input: Tensor) -> Tensor:
    floored = deepcopy(input)
    floored.name += ' (floored)'
    floored.value = np_floor(input.value)

    return floored


# ====================================================================================================
