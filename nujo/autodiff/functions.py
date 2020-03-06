from nujo.autodiff.function import Function
from nujo.autodiff.misc import matrix_dotprod_differentiation

__all__ = [
    'Addition',
    'Negation',
    'Multiplication',
    'Reciprocal',
    'Power',
    'MatrixMultiplication',
]

# ===================================================================================================


class Addition(Function):
    def __init__(self, input_a, input_b, name='<Add>'):
        super(Addition, self).__init__(input_a, input_b, name=name)

    def forward(self):
        return self.children[0].value + self.children[1].value

    def backward(self):
        return 1, 1


# ===================================================================================================


class Negation(Function):
    def __init__(self, input, name='<Neg>'):
        super(Negation, self).__init__(input, name=name)

    def forward(self):
        return -self.children[0].value

    def backward(self):
        return -1,


# ===================================================================================================


class Multiplication(Function):
    def __init__(self, input_a, input_b, name='<Mul>'):
        super(Multiplication, self).__init__(input_a, input_b, name=name)

    def forward(self):
        return self.children[0].value * self.children[1].value

    def backward(self):
        return self.children[1].value, self.children[0].value


# ===================================================================================================


class Reciprocal(Function):
    def __init__(self, input, name='<Recipr>'):
        super(Reciprocal, self).__init__(input, name=name)

    def forward(self):
        return 1 / (self.children[0].value + Reciprocal.epsilon)

    def backward(self):
        return -1 / ((self.children[0].value + Reciprocal.epsilon)**2),


# ===================================================================================================


class Power(Function):
    def __init__(self, input_a, input_b, name='<Pow>'):
        super(Power, self).__init__(input_a, input_b, name=name)

    def forward(self):
        return self.children[0].value**self.children[1].value

    def backward(self):
        # TODO: FIX wrong partial - the second

        return (self.children[1].value *
                self.children[0].value**(self.children[1].value - 1), 1)


# ===================================================================================================


class MatrixMultiplication(Function):
    def __init__(self, input_a, input_b, name='<MatMul>'):
        super(MatrixMultiplication, self).__init__(input_a, input_b, name=name)

    def forward(self):
        assert self.children[0].shape[1] == self.children[1].shape[0]
        return self.children[0].value @ self.children[1].value

    def backward(self):
        dinput0, dinput1 = matrix_dotprod_differentiation(
            self.children[0].value, self.children[1].value)

        return dinput0, dinput1


# ===================================================================================================
