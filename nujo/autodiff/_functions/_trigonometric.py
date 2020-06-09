from numpy import cos, ndarray, sin, tan

from nujo.autodiff.function import Function

__all__ = [
    '_Sin',
    '_Cos',
    '_Tan',
]

# ====================================================================================================


class _Sin(Function):
    def forward(self) -> ndarray:
        return sin(self.children[0].value)

    def backward(self, idx: int, accum_grad: Function.T) -> Function.T:
        return accum_grad * cos(self.children[0].value)


# ====================================================================================================


class _Cos(Function):
    def forward(self) -> ndarray:
        return cos(self.children[0].value)

    def backward(self, idx: int, accum_grad: Function.T) -> Function.T:
        return accum_grad * -sin(self.children[0].value)


# ====================================================================================================


class _Tan(Function):
    def forward(self) -> ndarray:
        return tan(self.children[0].value)

    def backward(self, idx: int, accum_grad: Function.T) -> Function.T:
        return accum_grad * (1 / cos(self.children[0].value))**2


# ====================================================================================================
