from numpy import array

from nujo.autodiff.function import Function
from nujo.autodiff.misc import (generate_tensor_name,
                                matrix_dotprod_differentiation)
from nujo.autodiff.tensor import Tensor

__all__ = [
    'Addition',
    'Subtraction',
    'Multiplication',
    'TrueDivision',
    'Power',
    'MatrixMultiplication',
]

# ===================================================================================================


class Addition(Function):
    def __init__(self, input_a, input_b, name='<Add>'):
        super(Addition, self).__init__([input_a, input_b], name=name)

    def forward(self):
        return Tensor(self.children[0].value + self.children[1].value,
                      children=self.children,
                      name=generate_tensor_name(self.id, self.name))

    def backward(self):
        return array(1), array(1)


# ===================================================================================================


class Subtraction(Function):
    def __init__(self, input_a, input_b, name='<Sub>'):
        super(Subtraction, self).__init__([input_a, input_b], name=name)

    def forward(self):
        return Tensor(self.children[0].value - self.children[1].value,
                      children=self.children,
                      name=generate_tensor_name(self.id, self.name))

    def backward(self):
        return array(1), array(-1)


# ===================================================================================================


class Multiplication(Function):
    def __init__(self, input_a, input_b, name='<Mul>'):
        super(Multiplication, self).__init__([input_a, input_b], name=name)

    def forward(self):
        return Tensor(self.children[0].value * self.children[1].value,
                      children=self.children,
                      name=generate_tensor_name(self.id, self.name))

    def backward(self):
        return (self.children[1].value, self.children[0].value)


# ===================================================================================================


class TrueDivision(Function):
    def __init__(self, input_a, input_b, name='<TrueDiv>'):
        super(TrueDivision, self).__init__([input_a, input_b], name=name)

    def forward(self):
        return Tensor(self.children[0].value / self.children[1].value,
                      children=self.children,
                      name=generate_tensor_name(self.id, self.name))

    def backward(self):
        return ((1 / self.children[1].value),
                ((-self.children[0].value) / (self.children[1]**2)))


# ===================================================================================================


class Power(Function):
    def __init__(self, input_a, input_b, name='<Pow>'):
        super(Power, self).__init__([input_a, input_b], name=name)

    def forward(self):
        return Tensor(self.children[0].value**self.children[1].value,
                      children=self.children,
                      name=generate_tensor_name(self.id, self.name))

    def backward(self):
        return ((self.children[1].value *
                 self.children[0].value**(self.children[1].value - 1)),
                array(1))  # TODO: FIX (wrong partial)


# ===================================================================================================


class MatrixMultiplication(Function):
    def __init__(self, input_a, input_b, name='<MatMul>'):
        super(MatrixMultiplication, self).__init__([input_a, input_b],
                                                   name=name)

    def forward(self):
        assert self.children[0].shape[1] == self.children[1].shape[0]

        return Tensor(self.children[0].value @ self.children[1].value,
                      children=self.children,
                      name=generate_tensor_name(self.id, self.name))

    def backward(self):
        dinput0, dinput1 = matrix_dotprod_differentiation(
            self.children[0].value, self.children[1].value)

        return dinput0, dinput1


# ===================================================================================================
