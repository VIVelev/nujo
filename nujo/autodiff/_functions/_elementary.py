from numpy import log, ndarray, ones

from nujo._typing import Union, _numerical
from nujo.autodiff.function import Function
from nujo.autodiff.tensor import Tensor

__all__ = [
    '_Addition',
    '_Negation',
    '_Multiplication',
    '_Reciprocal',
    '_Power',
    '_Logarithm',
    '_MatrixMul',
]

# ====================================================================================================


class _Addition(Function):
    def __init__(self,
                 input_a: Union[Tensor, _numerical],
                 input_b: Union[Tensor, _numerical],
                 name='Add'):
        super(_Addition, self).__init__(input_a, input_b, name=name)

    def forward(self) -> ndarray:
        return self.children[0].value + self.children[1].value

    def backward(self) -> tuple:
        return 1, 1


# ====================================================================================================


class _Negation(Function):
    def __init__(self, input: Union[Tensor, _numerical], name='Neg'):
        super(_Negation, self).__init__(input, name=name)

    def forward(self) -> ndarray:
        return -self.children[0].value

    def backward(self) -> tuple:
        return -1,


# ====================================================================================================


class _Multiplication(Function):
    def __init__(self,
                 input_a: Union[Tensor, _numerical],
                 input_b: Union[Tensor, _numerical],
                 name='Mul'):
        super(_Multiplication, self).__init__(input_a, input_b, name=name)

    def forward(self) -> ndarray:
        return self.children[0].value * self.children[1].value

    def backward(self) -> tuple:
        return self.children[1].value, self.children[0].value


# ====================================================================================================


class _Reciprocal(Function):
    def __init__(self,
                 input: Union[Tensor, _numerical],
                 name='Recipr',
                 eps=1e-18):
        super(_Reciprocal, self).__init__(input, name=name)
        self.eps = eps

    def forward(self) -> ndarray:
        return 1 / (self.children[0].value + self.eps)

    def backward(self) -> tuple:
        return -1 / ((self.children[0].value + self.eps)**2),


# ====================================================================================================


class _Power(Function):
    def __init__(self,
                 input_a: Union[Tensor, _numerical],
                 input_b: Union[Tensor, _numerical],
                 name='Pow'):
        super(_Power, self).__init__(input_a, input_b, name=name)

    def forward(self) -> ndarray:
        return self.children[0].value**self.children[1].value

    def backward(self) -> tuple:
        # TODO: FIX wrong partial - the second

        return (self.children[1].value *
                self.children[0].value**(self.children[1].value - 1), 1)


# ====================================================================================================


class _Logarithm(Function):
    def __init__(self,
                 input_a: Union[Tensor, _numerical],
                 input_b: Union[Tensor, _numerical],
                 name='Log'):
        super(_Logarithm, self).__init__(input_a, input_b, name=name)

        assert (self.children[0] > 0).all()  # argument value limit
        assert (self.children[1] > 0).all()  # base value limit
        assert (self.children[1] != 0).all()  # base value limit

    def forward(self) -> ndarray:
        return log(self.children[0].value) / log(self.children[1].value)

    def backward(self) -> tuple:
        return 1 / (self.children[0].value * log(self.children[1].value)), 1


# ====================================================================================================


class _MatrixMul(Function):
    def __init__(self,
                 input_a: Union[Tensor, _numerical],
                 input_b: Union[Tensor, _numerical],
                 name='MatMul'):
        super(_MatrixMul, self).__init__(input_a, input_b, name=name)

    @staticmethod
    def differentiate(X: 'Tensor', W: 'Tensor') -> tuple:
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

    def forward(self) -> ndarray:
        assert self.children[0].shape[-1] == self.children[1].shape[0]
        return self.children[0].value @ self.children[1].value

    def backward(self) -> tuple:
        return _MatrixMul.differentiate(*self.children)


# ====================================================================================================
