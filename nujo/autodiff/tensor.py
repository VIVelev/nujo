from copy import deepcopy

from numpy import array, eye, ndarray, tile

from nujo._typing import Union, _numerical
from nujo.autodiff import modes
from nujo.autodiff._node import _Node
from nujo.autodiff._utils import _if_not_none


class Tensor(_Node):
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
                 value: Union['Tensor', _numerical],
                 diff=True,
                 creator=None,
                 name='Tensor'):

        super(Tensor, self).__init__(*_if_not_none(creator), name=name)

        self.value: ndarray = value.value if isinstance(
            value, Tensor) else array(value)
        self.diff = diff
        self.creator = creator

        # (Tensor, weight) pair, used to backpropagate through the network
        # See: `Chain Rule` Wikipedia page for more info
        self._grad_dependencies = []

        # Gradient cache
        self._grad: 'Tensor' = None

        # Transposed tensor cache
        self._T: 'Tensor' = None

    @property
    def grad(self) -> 'Tensor':
        if self._grad is None:
            self._compute_grad()

        return self._grad

    @grad.setter
    def grad(self, value: Union['Tensor', _numerical]):
        self._grad = value if isinstance(value, Tensor) else Tensor(
            value, name=f'grad[{self.name}]')

    @grad.deleter
    def grad(self):
        del self._grad

    @property
    def T(self) -> 'Tensor':
        if self._T is None:
            transposed = deepcopy(self)
            transposed.value = self.value.T

            self._T = transposed

        return self._T

    # Shape and shape transformations

    @property
    def shape(self):
        return self.value.shape

    def reshape(self, *shape: int, inplace=False) -> 'Tensor':
        reshaped = self if inplace else deepcopy(self)
        reshaped.value = self.value.reshape(shape)
        return reshaped

    def repeat(self,
               *repeats: int,
               axis: int = None,
               inplace=False) -> 'Tensor':

        repeated = self if inplace else deepcopy(self)
        repeated.value = self.value.repeat(repeats, axis=axis)
        return repeated

    def squeeze(self, dim=-1, inplace=False) -> 'Tensor':
        if dim < 0:
            num_dims = len(self.shape)

            if dim < -num_dims:
                dim = num_dims
            else:
                dim += num_dims

        return self.reshape(*self.shape[:dim],
                            *self.shape[dim + 1:],
                            inplace=inplace)

    def unsqueeze(self, dim=-1, inplace=False) -> 'Tensor':
        if dim < 0:
            num_dims = len(self.shape)

            if dim < -num_dims:
                dim = 0
            else:
                if dim == -1:
                    dim += 1
                dim += num_dims

        return self.reshape(*self.shape[:dim],
                            1,
                            *self.shape[dim:],
                            inplace=inplace)

    # Gradient computation

    def add_grad_dependency(self, wrt: 'Tensor', weight: 'Tensor') -> None:
        self._grad_dependencies.append((wrt, weight))

    def _compute_grad(self, _debug=False) -> None:
        if modes.DIFF_ENABLED and self.diff and self._grad is None:
            if _debug:
                print()
                print('=' * 30)
                print(self, end='\n\n')
                print('Shape:', self.shape)
                print(f'Has {len(self._grad_dependencies)} dependencies:')
                print('Grad Dependecies:', self._grad_dependencies, end='\n\n')

            # Top-parent grad
            if len(self._grad_dependencies) == 0:
                self._grad = Tensor(1, name=f'grad[{self.name}]')
                return

            self._grad = Tensor(0, name=f'grad[{self.name}]')
            for z, weight in self._grad_dependencies:
                if _debug:
                    print('~' * 10)
                    print('Z Grad:', z.grad)
                    print('Shape:', z.grad.shape, end='\n\n')
                    print('-' * 5)
                    print('Z Weight:', weight)
                    print('Shape:', weight.shape, end='\n\n')

                if weight.shape == () or z.grad.shape == () or (
                        weight.shape == z.grad.shape):  # Is scalar
                    self._accumulate_grad_scalar(z, weight)
                else:
                    self._accumulate_grad_matrix(z, weight)

            if _debug:
                print('#' * 10)
                print('Current Grad:', self._grad)
                print('Shape:', self._grad.shape)
                print('-' * 5, end='\n\n')

    def _accumulate_grad_scalar(self, z: 'Tensor', weight: 'Tensor') -> None:
        self._grad.value = self._grad.value + z.grad.value * weight.value

    def _accumulate_grad_matrix(self, z: 'Tensor', weight: 'Tensor') -> None:
        weight.value = weight.value.reshape(z.grad.shape[0], self.shape[0],
                                            z.grad.shape[1] * self.shape[1])
        z_grad = z.grad.value.repeat(self.shape[1],
                                     axis=1).reshape(z.grad.shape[0], 1, -1)

        sum_mask = tile(eye(self.shape[1]), z.grad.shape[1])
        accumulated_grad = ((weight.value * z_grad) @ sum_mask.T).sum(0)

        self._grad.value = self.grad.value + accumulated_grad / z.grad.shape[0]

    def zero_grad(self) -> None:
        # `zero_grad` is called after an iteration.
        # The value of weight tensors is updated after an iteration.

        self._grad_dependencies = []
        self._grad = None
        self._T = None

    def backward(self, _debug=False) -> None:
        ''' It uses Breadth First Search to traverse the computation graph
        and compute the gradient for each differentiable Tensor in the graph.
        '''

        nodes_to_visit = [self]
        if _debug:
            i = 1

        while nodes_to_visit:
            node = nodes_to_visit.pop()
            node._compute_grad(_debug=_debug)

            if _debug:
                nstr = f' [{i}]'
                node.name += nstr if nstr not in node.name else ''
                i += 1

            if node.creator:
                for child in node.creator.children:
                    nodes_to_visit.insert(0, child)

    # Useful methods

    def all(self) -> ndarray:
        return self.value.all()

    def any(self) -> ndarray:
        return self.value.any()

    def __getitem__(self, position):
        return self.value[position]

    def __setitem__(self, position, value):
        self.value[position] = value

    def __hash__(self):
        return hash(self.name)

    # Static evaluation operator

    def __ilshift__(self, other):
        ''' In-place assignment operator: `<<=`

        Essentially used to achieve static evaluation.

        '''

        self.children = getattr(other, 'children', None)
        if self.children:
            try:
                self.children.remove(self)
            except ValueError:  # self is not in children
                pass

        self.creator = getattr(other, 'creator', None)
        if self.creator:
            try:
                self.creator.children.remove(self)
            except ValueError:  # self is not in children
                pass

        self.value = getattr(other, 'value', other)

        return self

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
        from nujo.autodiff._functions._elementary import _Addition
        return _Addition(self, other)()

    def __radd__(self, other):
        return self.__add__(other)

    def __neg__(self):
        from nujo.autodiff._functions._elementary import _Negation
        return _Negation(self)()

    def __sub__(self, other):
        return self.__add__(other.__neg__())

    def __rsub__(self, other):
        return self.__neg__().__add__(other)

    def __mul__(self, other):
        from nujo.autodiff._functions._elementary import _Multiplication
        return _Multiplication(self, other)()

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        from nujo.autodiff._functions._elementary import _Reciprocal
        return self.__mul__(_Reciprocal(other)())

    def __rtruediv__(self, other):
        from nujo.autodiff._functions._elementary import _Reciprocal
        return _Reciprocal(self)().__mul__(other)

    def __pow__(self, other):
        from nujo.autodiff._functions._elementary import _Power
        return _Power(self, other)()

    def __rpow__(self, other):
        from nujo.autodiff._functions._elementary import _Power
        return _Power(other, self)()

    # More complex arithmetic operations

    def __matmul__(self, other):
        from nujo.autodiff._functions._elementary import _MatrixMul
        return _MatrixMul(self, other)()

    def __rmatmul__(self, other):
        from nujo.autodiff._functions._elementary import _MatrixMul
        return _MatrixMul(other, self)()

    # Representations

    def __str__(self):
        return self.__repr__() + '\n' + '-' * 32 + '\n' + str(self.value)
