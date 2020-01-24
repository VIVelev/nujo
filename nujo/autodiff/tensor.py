from abc import abstractmethod
from copy import deepcopy

from numpy import array

from nujo.autodiff.constant import Constant
from nujo.autodiff.functions import (Addition, MatrixMultiplication,
                                     Multiplication, Power, Subtraction,
                                     TrueDivision)
from nujo.autodiff.misc import counter


class Tensor:
    ''' Tensor - multi-dimensional array

    Tensor is the basic block of computation in Nujo.

    Parameters:
    -----------
    value : value
    name : string
    children : list

    '''

    # Counter used to indicate the order of computations.
    z_counter = counter()

    def __init__(self, value, name='undefined', children=[]):
        self.value = value
        self.name = name
        self.children = children

        self.dependencies = []

        self._grad = None

    @property
    def grad(self):
        if self._grad is None:
            self.compute_grad()
        return self._grad

    @property
    def T(self):
        transposed = deepcopy(self)
        transposed.value = self.value.T
        return transposed

    @property
    def shape(self):
        self.value = array(self.value)
        return self.value.shape

    @abstractmethod
    def compute_grad(self):  # To be overridden by subclasses.
        pass

    def zero_grad(self):
        self.dependencies = []
        self._grad = None

    def backward(self):
        self.compute_grad()
        for child in self.children:
            child.backward()

        Tensor.z_counter.reset()  # A new forward pass awaits.

    def __add__(self, other):
        if not isinstance(other, Tensor):
            other = Constant(other)

        return Addition(self, other)()

    def __radd__(self, other):
        if not isinstance(other, Tensor):
            other = Constant(other)

        return other.__add__(self)

    def __sub__(self, other):
        if not isinstance(other, Tensor):
            other = Constant(other)

        return Subtraction(self, other)()

    def __rsub__(self, other):
        if not isinstance(other, Tensor):
            other = Constant(other)

        return other.__sub__(self)

    def __mul__(self, other):
        if not isinstance(other, Tensor):
            other = Constant(other)

        return Multiplication(self, other)()

    def __rmul__(self, other):
        if not isinstance(other, Tensor):
            other = Constant(other)

        return other.__mul__(self)

    def __truediv__(self, other):
        if not isinstance(other, Tensor):
            other = Constant(other)

        return TrueDivision(self, other)()

    def __rtruediv__(self, other):
        if not isinstance(other, Tensor):
            other = Constant(other)

        return other.__truediv__(self)

    def __pow__(self, other):
        if not isinstance(other, Tensor):
            other = Constant(other)

        return Power(self, other)()

    def __matmul__(self, other):
        assert self.shape[1] == other.shape[0]

        if not isinstance(other, Tensor):
            other = Constant(other)

        return MatrixMultiplication(self, other)()

    def __rmatmul__(self, other):
        if not isinstance(other, Tensor):
            other = Constant(other)

        return other.__matmul__(self)

    def __repr__(self):
        return self.name
