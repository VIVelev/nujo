from numbers import Number
from typing import List, Tuple, Union

from numpy import array, empty, ndarray

import nujo.autodiff.modes as modes
from nujo.autodiff._node import _Node
from nujo.autodiff._utils import _if_not_none


class Tensor(_Node):
    ''' Tensor - a multi-dimensional array

    Tensors are the main units of data in nujo.
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

        self._value: ndarray = None
        self.value = value  # set value

        self.diff = diff
        self.creator = creator

        # Outputs of the functions the current tensor is input to.
        # Used for backpropagation of the gradients.
        self.parents_outputs: List['Tensor'] = []

        # Gradient of the current tensor
        self._grad: 'Tensor' = None

        # Transposed tensor cache
        self._T: 'Tensor' = None
        self._prev_value: ndarray = None

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value: Union['Tensor', ndarray, List[Number], Number]):
        if isinstance(value, Tensor):
            self._value = value.value
        elif isinstance(value, ndarray):
            self._value = value
        else:
            self._value = array(value)

    @value.deleter
    def value(self):
        del self._value

    @property
    def grad(self) -> 'Tensor':
        if self._grad is None:
            self._grad = Tensor(empty(self._value.shape),
                                name=f'grad[{self.name}]')

        return self._grad

    # Shape and shape manipulations

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._value.shape

    @property
    def T(self) -> 'Tensor':
        # Only transpose if something has changed
        if (self._value != self._prev_value).any():
            self._T = self.transpose()
            self._prev_value = self._value

        return self._T

    def transpose(self, *dims: int) -> 'Tensor':
        from nujo.autodiff._functions._transform import _Transpose
        return _Transpose(self, dims)()

    def reshape(self, *shape: int) -> 'Tensor':
        from nujo.autodiff._functions._transform import _Reshape
        return _Reshape(self, shape)()

    def squeeze(self, dim=-1) -> 'Tensor':
        if dim < 0:
            num_dims = len(self._value.shape)

            if dim < -num_dims:
                dim = num_dims
            else:
                dim += num_dims

        return self.reshape(*self._value.shape[:dim],
                            *self._value.shape[dim + 1:])

    def unsqueeze(self, dim=-1) -> 'Tensor':
        if dim < 0:
            num_dims = len(self._value.shape)

            if dim < -num_dims:
                dim = 0
            else:
                if dim == -1:
                    dim += 1
                dim += num_dims

        return self.reshape(*self._value.shape[:dim], 1,
                            *self._value.shape[dim:])

    # Gradient computation

    def _compute_grad_from(self,
                           poutput: 'Tensor') -> Union['Tensor', ndarray]:
        ''' Computes the gradient of `self` w.r.t. the output of the computation
        graph from `poutput` (using the path of computations from `poutput`)

            In other words, this functions returns:
                (dOutput / dPoutput) * (dPoutput / dSelf)

        '''

        # Find the index of the children which gradient should be computed
        # (a.k.a. find the index of `self` in `poutput.creator.children`)
        idx = next(i for i, v in enumerate(poutput.creator.children)
                   if v is self)

        if poutput._grad.diff:
            # Pass a diff enabled tensor to the backward call,
            # thus recording grad computations in the computation
            # graph, which enables higher-order differentiation.
            grad = poutput.creator.backward(idx, poutput._grad)

            # Check if `self` is scalar and needs to be averaged
            if self._value.shape != () and\
               self._value.shape[-1] == 1:

                # Record the mean in the computation graph
                from nujo.math.aggregate import mean
                grad = mean(grad, dim=-1, keepdim=True)

        else:
            # Do not leave a trace in the computation graph!
            # Use numpy arrays! :)
            grad = poutput.creator.backward(idx, poutput._grad._value)

            # Check if `self` is scalar and needs to be averaged
            if self._value.shape != () and\
               self._value.shape[-1] == 1:

                grad = grad.mean(axis=-1, keepdims=True)

        return grad

    def compute_grad(self) -> None:
        if modes.DIFF_ENABLED and self.diff:

            # Make sure grad is Tensor (`grad property call`) and init value
            if self._grad is None:
                self.zero_grad(propagate=False)

            # Top-parent grad
            if len(self.parents_outputs) == 0:
                self._grad._value += 1
                return

            for poutput in self.parents_outputs:
                curr_grad = self._compute_grad_from(poutput)

                if self._grad.diff:
                    # Record grad computations in the computation graph
                    self._grad += curr_grad
                else:
                    self._grad._value += curr_grad

    def zero_grad(self, propagate=True) -> None:
        self.grad._value.fill(0)

        if propagate:
            for poutput in self.parents_outputs:
                poutput.zero_grad()

    def backward(self, _debug=False) -> None:
        ''' It uses Breadth First Search to traverse the computation graph
        and compute the gradient for each differentiable Tensor in the graph.

        '''

        nodes_to_visit: List['Tensor'] = [self]
        if _debug:
            i = 1

        while nodes_to_visit:
            node = nodes_to_visit.pop()
            node.compute_grad()

            if _debug:
                nstr = f' [{i}]'
                node.name += nstr if nstr not in node.name else ''
                i += 1

            if node.creator:
                for child in node.creator.children:
                    # Avoid visiting the same node twice
                    if all(child is not node for node in nodes_to_visit):
                        nodes_to_visit.insert(0, child)

    # Useful methods

    def all(self) -> ndarray:
        return self._value.all()

    def any(self) -> ndarray:
        return self._value.any()

    def __getitem__(self, position: Union[int, Tuple[int, ...]]):
        return Tensor(self._value[position],
                      diff=self.diff,
                      creator=self.creator,
                      name=f'{self.name}[{position}]')

    def __setitem__(self, position: Union[int, Tuple[int, ...]],
                    value: Union['Tensor', ndarray, List[Number], Number]):

        # TODO: This is a naive implementation. Fix it.
        self._value[position] = value

    def __hash__(self):
        return self.id

    # Static evaluation operator

    def __ilshift__(
            self, other: Union['Tensor', ndarray, List[Number],
                               Number]) -> 'Tensor':
        ''' In-place assignment operator: `<<=`

        Transfering key properties from `other` to `self`.
        Essentially a shortcut for:
            >>> self.children = other.children
            >>> self.creator = other.creator
            >>> self.value = other.value
            >>> self.grad = other.grad

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

        # Transfer the gradient
        self._grad = getattr(other, 'grad', None)

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
        # TODO: Come up with a better representation
        return self.__repr__() + '\n' + '-' * 32 + '\n' + str(self._value)
