from nujo.autodiff import Tensor
from nujo.autodiff.functions import (Addition, MatrixMul, Multiplication,
                                     Negation, Power, Reciprocal)

__all__ = [
    'add',
    'neg',
    'mul',
    'recipr',
    'pow',
    'matmul',
]

# ====================================================================================================


def add(input_a: Tensor, input_b: Tensor) -> Tensor:
    return Addition(input_a, input_b)()


def neg(input: Tensor) -> Tensor:
    return Negation(input)()


def mul(input_a: Tensor, input_b: Tensor) -> Tensor:
    return Multiplication(input_a, input_b)()


def recipr(input: Tensor) -> Tensor:
    return Reciprocal(input)()


def pow(input_a: Tensor, input_b: Tensor) -> Tensor:
    return Power(input_a, input_b)()


def matmul(input_a: Tensor, input_b: Tensor) -> Tensor:
    return MatrixMul(input_a, input_b)()


# ====================================================================================================
