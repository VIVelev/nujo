from numpy import array, eye, tile

from nujo.autodiff.modes import DIFF_ENABLED
from nujo.autodiff.node import Node


class Tensor(Node):
    ''' Tensor - multi-dimensional array

    Tensor is the basic block of computation in Nujo.

    Parameters:
    -----------
    value : value, numerical value of the tensor
    children : varargs, the tensors form which this tensor is produced
    name : string, representation of the tensor

    '''
    def __init__(self, value, diff=True, *children, name='<Tensor>'):
        super(Tensor, self).__init__(*children, name=name)

        self.value = array(value)
        self.diff = diff

        self.grad_dependencies = []

        # Gradient cache
        self._grad = None

        # Transposed tensor cache
        self._T = None

    @property
    def grad(self):
        if self._grad is None:
            self.compute_grad()

        return self._grad

    @property
    def T(self):
        if self._T is None:
            from copy import deepcopy
            transposed = deepcopy(self)
            transposed.value = self.value.T

            self._T = transposed

        return self._T

    @property
    def shape(self):
        return self.value.shape

    def add_grad_dependency(self, wrt, weight):
        self.grad_dependencies.append((wrt, weight))

    def compute_grad(self, debug=False):
        if self.diff and DIFF_ENABLED:
            if debug:
                print()
                print('=' * 30)
                print(self, self.shape, '- dependencies')

            if len(self.grad_dependencies) == 0:
                self._grad = array(1)
                return

            self._grad = 0
            for z, weight in self.grad_dependencies:
                if debug:
                    print('-' * 10)
                    print('Weight of `Z_prev Grad`:', weight)
                    print('Shape:', weight.shape)
                    print('-' * 5)
                    print('Z_prev Grad:', z.grad)
                    print('Shape:', z.grad.shape)
                    print('-' * 5)

                if weight.shape == () or z.grad.shape == ():
                    self._grad += weight * z.grad
                else:
                    weight = weight.reshape(z.grad.shape[0], self.shape[0],
                                            z.grad.shape[1] * self.shape[1])
                    z_grad = z.grad.repeat(self.shape[1], axis=1).reshape(
                        z.grad.shape[0], 1, -1)
                    sum_mask = tile(eye(self.shape[1]), z.grad.shape[1])
                    accumulated_grad = ((weight * z_grad) @ sum_mask.T).sum(0)
                    self._grad += accumulated_grad / z.grad.shape[0]

            if debug:
                print('Current Grad:', self._grad)
                print('Shape:', self._grad.shape)
                print('-' * 5)
                print()

    def zero_grad(self):
        # `zero_grad` is called after an iteration.
        # The value of weight tensors is updated after an iteration.

        self.grad_dependencies = []
        self._grad = None
        self._T = None

    def backward(self):
        self.compute_grad()
        for child in self.children:
            child.backward()

    def __add__(self, other):
        from nujo.autodiff.functions import Addition
        return Addition(self, other)()

    def __radd__(self, other):
        return self.__add__(other)

    def __neg__(self):
        from nujo.autodiff.functions import Negation
        return Negation(self)()

    def __sub__(self, other):
        return self.__add__(other.__neg__())

    def __rsub__(self, other):
        return self.__neg__().__add__(other)

    def __mul__(self, other):
        from nujo.autodiff.functions import Multiplication
        return Multiplication(self, other)()

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        from nujo.autodiff.functions import Reciprocal
        return self.__mul__(Reciprocal(other))

    def __rtruediv__(self, other):
        from nujo.autodiff.functions import Reciprocal
        return Reciprocal(self).__mul__(other)

    def __pow__(self, other):
        from nujo.autodiff.functions import Power
        return Power(self, other)()

    def __matmul__(self, other):
        from nujo.autodiff.functions import MatrixMultiplication
        return MatrixMultiplication(self, other)()

    def __rmatmul__(self, other):
        from nujo.autodiff.functions import MatrixMultiplication
        return MatrixMultiplication(other, self)()

    def __repr__(self):
        return self.name
