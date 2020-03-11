from __future__ import annotations

from typing import Any

from numpy import array, eye, ndarray, tile

from nujo.autodiff.modes import DIFF_ENABLED
from nujo.autodiff.node import Node
from nujo.autodiff.utils import if_not_none


class Tensor(Node):
    ''' Tensor - a multi-dimensional array

    Tensors are the main units of data and computation in Nujo.
    They "flow" in the computation graph. :)

    Tensors can be either constants or trainable weights,
    depending on whether gradients are computed for the given tensor.

    Parameters:
    -----------
    value : value, numerical value of the tensor
    diff : boolean, whether to compute gradients for the tensor
    creator : Nujo Function, that created this tensor;
    the only child of a tensor
    name : string, representation of the tensor

    '''
    def __init__(self,
                 value: Any,
                 diff=True,
                 creator: Node = None,
                 name='<Tensor>') -> None:

        super(Tensor, self).__init__(*if_not_none(creator), name=name)

        self.value = array(value)
        self.diff = diff
        self.creator = creator

        # (Tensor, weight) pair, used to backpropagate through the network
        # See: `Chain Rule` Wikipedia page for more info
        self._grad_dependencies: list[tuple['Tensor', ndarray]] = []

        # Gradient cache
        self._grad: ndarray = None

        # Transposed tensor cache
        self._T: ndarray = None

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

    def add_grad_dependency(self, wrt: 'Tensor', weight: ndarray) -> None:
        self._grad_dependencies.append((wrt, weight))

    def compute_grad(self, debug=False) -> None:
        if self.diff and DIFF_ENABLED:
            if debug:
                print()
                print('=' * 30)
                print(self, self.shape, '- dependencies')

            if len(self._grad_dependencies) == 0:
                self._grad = array(1)
                return

            self._grad = 0
            for z, weight in self._grad_dependencies:
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

    def zero_grad(self) -> None:
        # `zero_grad` is called after an iteration.
        # The value of weight tensors is updated after an iteration.

        self._grad_dependencies = []
        self._grad = None
        self._T = None

    def backward(self) -> None:
        self.compute_grad()

        if self.creator:
            for child in self.creator.children:
                child.backward()

    # Useful methods

    def all(self) -> ndarray:
        return self.value.all()

    def any(self) -> ndarray:
        return self.value.any()

    def __getitem__(self, position):
        return self.value[position]

    # Comparison operations

    def __lt__(self, other):
        return self.value < getattr(other, 'value', other)

    def __le__(self, other):
        return self.value <= getattr(other, 'value', other)

    def __eq__(self, other):
        return self.value == getattr(other, 'value', other)

    def __ne__(self, other):
        return self.value != getattr(other, 'value', other)

    def __gt__(self, other):
        return self.value > getattr(other, 'value', other)

    def __ge__(self, other):
        return self.value >= getattr(other, 'value', other)

    # Arithmetic operations

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
        return self.__mul__(Reciprocal(other)())

    def __rtruediv__(self, other):
        from nujo.autodiff.functions import Reciprocal
        return Reciprocal(self)().__mul__(other)

    def __pow__(self, other):
        from nujo.autodiff.functions import Power
        return Power(self, other)()

    # More complex arithmetic operations

    def __matmul__(self, other):
        from nujo.autodiff.functions import MatrixMul
        return MatrixMul(self, other)()

    def __rmatmul__(self, other):
        from nujo.autodiff.functions import MatrixMul
        return MatrixMul(other, self)()

    # Representation

    def __repr__(self):
        return self.name
