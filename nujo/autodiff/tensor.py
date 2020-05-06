from copy import copy
from numbers import Number
from typing import List, Optional, Tuple, Union

from numpy import array, ndarray, ones, zeros

from nujo.autodiff import modes
from nujo.autodiff._node import _Node
from nujo.autodiff._utils import _if_not_none


class Tensor(_Node):
    ''' Tensor - a multi-dimensional array

    Tensors are the main units of data and computation in nujo.
    They "flow" in the computation graph. :)

    Tensors can be either constants or trainable weights,
    depending on whether gradients are computed for the given tensor.

    Parameters:
    -----------
     - value : value, numerical value of the tensor
     - diff : boolean, whether to compute gradients for the tensor
     - creator : nujo function, that created this tensor;
       the only child of a tensor
     - name : string, representation of the tensor

    '''
    def __init__(self,
                 value: Union['Tensor', ndarray, List[Number], Number],
                 diff=False,
                 creator=None,
                 name='Tensor'):

        super(Tensor, self).__init__(*_if_not_none(creator), name=name)

        self._value: ndarray = value.value if isinstance(
            value, Tensor) else array(value)

        self.diff = diff
        self.creator = creator

        # (Tensor, weight) pair, used to backpropagate through the network
        # See: `Chain Rule` Wikipedia page for more info
        self.backward_depend: List[List['Tensor', ndarray]] = []

        # Gradient cache
        self._grad: 'Tensor' = None

        # Transposed tensor cache
        self._T: 'Tensor' = None

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value: Union['Tensor', ndarray, List[Number], Number]):
        self._value = value.value if isinstance(value,
                                                Tensor) else array(value)

    @value.deleter
    def value(self):
        del self._value

    @property
    def grad(self) -> 'Tensor':
        if not isinstance(self._grad, Tensor):
            self._grad = Tensor(None, name=f'grad[{self.name}]')

        if self._grad.value is None:
            self._compute_grad()

        return self._grad

    @grad.setter
    def grad(self, value: Union['Tensor', ndarray, List[Number], Number]):
        if not isinstance(self._grad, Tensor):
            self._grad = Tensor(None, name=f'grad[{self.name}]')

        self._grad.value = value

    @grad.deleter
    def grad(self):
        del self._grad

    @property
    def T(self) -> 'Tensor':
        if not isinstance(self._T, Tensor):
            self._T = copy(self)
            self._T.value = None

        if self._T.value is None:
            self._T.value = self._value.T

        return self._T

    # Shape and shape transformations

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._value.shape

    # TODO: Inplace?
    def reshape(self, *shape: int, inplace=False) -> 'Tensor':
        reshaped = self if inplace else copy(self)
        reshaped.value = self._value.reshape(shape)
        return reshaped

    def repeat(self,
               *repeats: int,
               axis: Optional[int] = None,
               inplace=False) -> 'Tensor':

        repeated = self if inplace else copy(self)
        repeated.value = self._value.repeat(repeats, axis=axis)
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

    def add_backward_dep(self, wrt: 'Tensor', weight: ndarray) -> None:
        self.backward_depend.append([wrt, weight])

    def _intersect_parents(self, *others: 'Tensor') -> Optional['Tensor']:
        common_parents = set([dep[0] for dep in self.backward_depend])

        for other in others:
            if isinstance(other, Tensor):
                common_parents.intersection_update(
                    set([dep[0] for dep in other.backward_depend]))

        return common_parents

    def _compute_grad(self, _debug=False) -> None:
        if modes.DIFF_ENABLED and self.diff:
            if not isinstance(self._grad, Tensor):
                self._grad = Tensor(None, name=f'grad[{self.name}]')

            if _debug:
                print()
                print('=' * 30)
                print(self, end='\n\n')
                print('Shape:', self.shape)
                print(f'Has {len(self.backward_depend)} dependencies:')
                print('Grad Dependecies:', self.backward_depend, end='\n\n')

            # Top-parent grad
            if len(self.backward_depend) == 0:
                self._grad.value = ones(self.shape)
                return

            self._grad.value = zeros(self.shape)
            for z, weight in self.backward_depend:
                if _debug:
                    print('~' * 10)
                    print('Z Grad:', z.grad)
                    print('Shape:', z.grad.shape, end='\n\n')
                    print('-' * 5)
                    print('Z Weight:', weight)
                    print('Shape:', weight.shape, end='\n\n')

                if z.creator.name == 'MatMul':
                    if self.id == z.creator.children[0].id:
                        # XW = Z, dX ...
                        self._grad.value += z.grad.value @ weight.T

                    else:
                        # XW = Z, dW ...
                        self._grad.value += (z.grad.value.T @ weight).T

                else:
                    self._grad.value = self._grad.value + \
                        z.grad.value * weight

            if _debug:
                print('#' * 10)
                print('Current Grad:', self._grad)
                print('Shape:', self._grad.shape)
                print('-' * 5, end='\n\n')

    def zero_grad(self) -> None:
        # `zero_grad` is called after an iteration.
        # The value of weight tensors is updated after an iteration.

        self.grad.value = None
        self.T.value = None

    def backward(self, _debug=False) -> None:
        ''' It uses Breadth First Search to traverse the computation graph
        and compute the gradient for each differentiable Tensor in the graph.

        '''

        nodes_to_visit: List['Tensor'] = [self]
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
        return self._value.all()

    def any(self) -> ndarray:
        return self._value.any()

    def __getitem__(self, position: Union[int, Tuple[int, ...]]):
        return self._value[position]

    def __setitem__(self, position: Union[int, Tuple[int, ...]],
                    value: Union['Tensor', ndarray, List[Number], Number]):

        self._value[position] = value

    def __hash__(self):
        return hash(self.name)

    # Static evaluation operator

    def __ilshift__(self, other: Union['Tensor', ndarray, List[Number],
                                       Number]):
        ''' In-place assignment operator: `<<=`

        Transfering key properties from `other` to `self`.
        Essentially a shortcut for:
            >>> self.children = other.children
            >>> self.creator = other.creator
            >>> self.value = other.value

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

        self._value = getattr(other, 'value', other)

        return self

    # Comparison operations

    def __lt__(self, other):
        return self._value < getattr(other, 'value', other)

    def __le__(self, other):
        return self._value <= getattr(other, 'value', other)

    def __eq__(self, other):
        return self._value == getattr(other, 'value', other)

    def __ne__(self, other):
        return self._value != getattr(other, 'value', other)

    def __gt__(self, other):
        return self._value > getattr(other, 'value', other)

    def __ge__(self, other):
        return self._value >= getattr(other, 'value', other)

    # Arithmetic operations

    def __add__(self, other):
        from nujo.autodiff._functions._elementary import _Addition

        outputs = self._intersect_parents(other)
        for output in outputs:
            if isinstance(output.creator, _Addition):
                return output.creator()

        return _Addition(self, other)()

    def __radd__(self, other):
        return self.__add__(other)

    def __neg__(self):
        from nujo.autodiff._functions._elementary import _Negation

        outputs = self._intersect_parents()
        for output in outputs:
            if isinstance(output.creator, _Negation):
                return output.creator()

        return _Negation(self)()

    def __sub__(self, other):
        return self.__add__(other.__neg__())

    def __rsub__(self, other):
        return self.__neg__().__add__(other)

    def __mul__(self, other):
        from nujo.autodiff._functions._elementary import _Multiplication

        outputs = self._intersect_parents(other)
        for output in outputs:
            if isinstance(output.creator, _Multiplication):
                return output.creator()

        return _Multiplication(self, other)()

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        from nujo.autodiff._functions._elementary import _Reciprocal

        if isinstance(other, Tensor):
            outputs = other._intersect_parents()
            for output in outputs:
                if isinstance(output.creator, _Reciprocal):
                    return self.__mul__(output.creator())

        return self.__mul__(_Reciprocal(other)())

    def __rtruediv__(self, other):
        from nujo.autodiff._functions._elementary import _Reciprocal

        outputs = self._intersect_parents()
        for output in outputs:
            if isinstance(output.creator, _Reciprocal):
                return output.creator().__mul__(other)

        return _Reciprocal(self)().__mul__(other)

    def __pow__(self, other):
        from nujo.autodiff._functions._elementary import _Power

        outputs = self._intersect_parents(other)
        for output in outputs:
            if isinstance(output.creator, _Power) and \
               output.creator.children[0] is self:

                return output.creator()

        return _Power(self, other)()

    def __rpow__(self, other):
        from nujo.autodiff._functions._elementary import _Power

        outputs = self._intersect_parents(other)
        for output in outputs:
            if isinstance(output.creator, _Power) and \
               output.creator.children[1] is self:

                return output.creator()

        return _Power(other, self)()

    # More complex arithmetic operations

    def __matmul__(self, other):
        from nujo.autodiff._functions._elementary import _MatrixMul

        outputs = self._intersect_parents(other)
        for output in outputs:
            if isinstance(output.creator, _MatrixMul) and \
               output.creator.children[0] is self:

                return output.creator()

        return _MatrixMul(self, other)()

    def __rmatmul__(self, other):
        from nujo.autodiff._functions._elementary import _MatrixMul

        outputs = self._intersect_parents(other)
        for output in outputs:
            if isinstance(output.creator, _MatrixMul) and \
               output.creator.children[1] is self:

                return output.creator()

        return _MatrixMul(other, self)()

    # Representations

    def __str__(self):
        # TODO: Come up with a better representation
        return self.__repr__() + '\n' + '-' * 32 + '\n' + str(self._value)
