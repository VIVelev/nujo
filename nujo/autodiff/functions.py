from nujo.autodiff.function import Function
from nujo.autodiff.misc import (generate_tensor_name,
                                matrix_dotprod_differentiation)
from nujo.autodiff.modes import DIFF_ENABLED
from nujo.autodiff.variable import Variable

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
    def __init__(self, input_a, input_b, name='Add'):
        super(Addition, self).__init__(input_a, input_b, name=name)

    def forward(self):
        return Variable(self.inputs[0].value + self.inputs[1].value,
                        name=generate_tensor_name(
                            self.inputs[0].z_counter.get(), self.name),
                        children=self.inputs if DIFF_ENABLED else [])

    def backward(self, grad):
        return grad, grad


# ===================================================================================================


class Subtraction(Function):
    def __init__(self, input_a, input_b, name='Sub'):
        super(Subtraction, self).__init__(input_a, input_b, name=name)

    def forward(self):
        return Variable(self.inputs[0].value - self.inputs[1].value,
                        name=generate_tensor_name(
                            self.inputs[0].z_counter.get(), self.name),
                        children=self.inputs if DIFF_ENABLED else [])

    def backward(self, grad):
        return grad, -grad


# ===================================================================================================


class Multiplication(Function):
    def __init__(self, input_a, input_b, name='Mul'):
        super(Multiplication, self).__init__(input_a, input_b, name=name)

    def forward(self):
        return Variable(self.inputs[0].value * self.inputs[1].value,
                        name=generate_tensor_name(
                            self.inputs[0].z_counter.get(), self.name),
                        children=self.inputs if DIFF_ENABLED else [])

    def backward(self, grad):
        return (self.inputs[1].value * grad, self.inputs[0].value * grad)


# ===================================================================================================


class TrueDivision(Function):
    def __init__(self, input_a, input_b, name='TrueDiv'):
        super(TrueDivision, self).__init__(input_a, input_b, name=name)

    def forward(self):
        return Variable(self.inputs[0].value / self.inputs[1].value,
                        name=generate_tensor_name(
                            self.inputs[0].z_counter.get(), self.name),
                        children=self.inputs if DIFF_ENABLED else [])

    def backward(self, grad):
        return ((1 / self.inputs[1].value) * grad,
                ((-self.inputs[0].value) / (self.inputs[1]**2)) * grad)


# ===================================================================================================


class Power(Function):
    def __init__(self, input_a, input_b, name='Pow'):
        super(Power, self).__init__(input_a, input_b, name=name)

    def forward(self):
        return Variable(self.inputs[0].value**self.inputs[1].value,
                        name=generate_tensor_name(
                            self.inputs[0].z_counter.get(), self.name),
                        children=self.inputs if DIFF_ENABLED else [])

    def backward(self, grad):
        return ((self.inputs[1].value *
                 self.inputs[0].value**(self.inputs[1].value - 1)) * grad, grad
                )  # TODO: FIX (wrong partial)


# ===================================================================================================


class MatrixMultiplication(Function):
    def __init__(self, input_a, input_b, name='MatMul'):
        super(MatrixMultiplication, self).__init__(input_a, input_b, name=name)

    def forward(self):
        return Variable(self.inputs[0].value @ self.inputs[1].value,
                        name=generate_tensor_name(
                            self.inputs[0].z_counter.get(), self.name),
                        children=self.inputs if DIFF_ENABLED else [])

    def backward(self, grad):
        dinput0, dinput1 = matrix_dotprod_differentiation(
            self.inputs[0].value, self.inputs[1].value)

        return dinput0 * grad, dinput1 * grad


# ===================================================================================================
