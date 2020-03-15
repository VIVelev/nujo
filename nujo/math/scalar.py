from nujo.autodiff import Tensor
from nujo.autodiff.functions import Power

__all__ = [
    'sqrt',
    'abs',
]

# ====================================================================================================


def sqrt(input: Tensor) -> Tensor:
    return Power(input, 1 / 2)()


def abs(input: Tensor) -> Tensor:
    return sqrt(input**2)


def ceil(input: Tensor) -> Tensor:
    pass


def floor(input: Tensor) -> Tensor:
    pass


# ====================================================================================================
