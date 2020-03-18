from numpy import log, ones

from nujo.autodiff.function import Function

__all__ = [
    'Addition',
    'Negation',
    'Multiplication',
    'Reciprocal',
    'Power',
    'Logarithm',
    'MatrixMul',
]

# ====================================================================================================


class Addition(Function):
    def __init__(self, input_a, input_b, name='<Add>'):
        super(Addition, self).__init__(input_a, input_b, name=name)

    def forward(self):
        return self.children[0].value + self.children[1].value

    def backward(self):
        return 1, 1


# ====================================================================================================


class Negation(Function):
    def __init__(self, input, name='<Neg>'):
        super(Negation, self).__init__(input, name=name)

    def forward(self):
        return -self.children[0].value

    def backward(self):
        return -1,


# ====================================================================================================


class Multiplication(Function):
    def __init__(self, input_a, input_b, name='<Mul>'):
        super(Multiplication, self).__init__(input_a, input_b, name=name)

    def forward(self):
        return self.children[0].value * self.children[1].value

    def backward(self):
        return self.children[1].value, self.children[0].value


# ====================================================================================================


class Reciprocal(Function):
    def __init__(self, input, name='<Recipr>'):
        super(Reciprocal, self).__init__(input, name=name)

    def forward(self):
        return 1 / (self.children[0].value + Reciprocal.epsilon)

    def backward(self):
        return -1 / ((self.children[0].value + Reciprocal.epsilon)**2),


# ====================================================================================================


class Power(Function):
    def __init__(self, input_a, input_b, name='<Pow>'):
        super(Power, self).__init__(input_a, input_b, name=name)

    def forward(self):
        return self.children[0].value**self.children[1].value

    def backward(self):
        # TODO: FIX wrong partial - the second

        return (self.children[1].value *
                self.children[0].value**(self.children[1].value - 1), 1)


# ====================================================================================================


class Logarithm(Function):
    def __init__(self, input_a, input_b, name='<Log>'):
        super(Logarithm, self).__init__(input_a, input_b, name=name)

        assert (self.children[0] > 0).all()  # argument value limit
        assert (self.children[1] > 0).all()  # base value limit
        assert (self.children[1] != 0).all()  # base value limit

    def forward(self):
        return log(self.children[0].value) / log(self.children[1].value)

    def backward(self):
        return 1 / (self.children[0].value * log(self.children[1].value)), 1


# ====================================================================================================


class MatrixMul(Function):
    def __init__(self, input_a, input_b, name='<MatMul>'):
        super(MatrixMul, self).__init__(input_a, input_b, name=name)

    @staticmethod
    def differentiate(X, W):
        ''' Calculate Matrix partial derivatives

        Given Z = XW, Calculate:
            - dX = dZ/dX
            - dW = dZ/dW

        Parameters:
        -----------
        X : matrix, left hand multiplier
        W : matrix, right hand multiplier

        Returns:
        --------
        dX : partial derivative of Z w.r.t. X
        dW : partial derivative of Z w.r.t. W

        '''

        # ------------------------------------------------------------
        dX = ones((X.shape[0]**2, W.shape[1] * X.shape[1]))

        i, j = 0, 0  # indecies of Z
        k, m = 0, 0  # indecies of X
        # p, q : indecies of dX

        for p in range(dX.shape[0]):
            for q in range(dX.shape[1]):
                if k == i:
                    dX[p, q] = W[m, j]

                j = q % W.shape[1]
                m = q % X.shape[1]

            i = q % X.shape[0]
            k = p % X.shape[0]

        # ------------------------------------------------------------
        dW = ones((X.shape[0] * W.shape[0], W.shape[1]**2))

        i, j = 0, 0  # indecies of Z
        k, m = 0, 0  # indecies of W
        # p, q : indecies of dW

        for p in range(dW.shape[0]):
            for q in range(dW.shape[1]):
                if m == j:
                    dW[p, q] = X[i, k]

                j = q % W.shape[1]
                m = q % W.shape[1]

            i = q % X.shape[0]
            k = p % W.shape[0]

        ##############################################################

        return dX, dW

    def forward(self):
        assert self.children[0].shape[1] == self.children[1].shape[0]
        return self.children[0].value @ self.children[1].value

    def backward(self):
        return MatrixMul.differentiate(*self.children)


# ====================================================================================================
