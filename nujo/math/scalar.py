from numpy import around as np_round
from numpy import ceil as np_ceil
from numpy import floor as np_floor

from nujo.autodiff import Tensor
from nujo.autodiff.functions import Power

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
    ''' In-place function '''
    np_round(input.value, out=input.value)
    return input


def ceil(input: Tensor) -> Tensor:
    ''' In-place function '''
    np_ceil(input.value, out=input.value)
    return input


def floor(input: Tensor) -> Tensor:
    ''' In-place function '''
    np_floor(input.value, out=input.value)
    return input


# ====================================================================================================
