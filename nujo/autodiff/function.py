from abc import abstractmethod
from numbers import Number
from typing import Dict, List, Tuple, Union

from numpy import ndarray

from nujo.autodiff import modes
from nujo.autodiff._node import _Node
from nujo.autodiff.tensor import Tensor


class Function(_Node, object):
    ''' Base Class for functions

    Functions are applied to tensors.

    A Function takes multiple tensors as input
    and produces only one tensor as output.

    Functions do not change tensors in-place.

    Parameters:
    -----------
     - children : varargs, the inpute tensors
     - name : string, the name of the function

    '''

    _children_history: Dict[Tuple[Union[Tensor, ndarray, List[Number], Number],
                                  ...], 'Function'] = {}
    _cache_hit = False

    def __init__(self,
                 *children: Union[Tensor, ndarray, List[Number], Number],
                 name='Function'):

        if self._cache_hit:
            return

        super(Function, self).__init__(*children, name=name)

        # This placeholder is reused when possible
        self._output_placeholder = Tensor(
            [[None]],
            diff=any([x.diff for x in self.children]) and modes.DIFF_ENABLED,
            creator=self if modes.DIFF_ENABLED else None,
            name=self._generate_tensor_name())

        if modes.DIFF_ENABLED:  # If graph building is enabled.
            for child in self.children:
                child.parents_outputs.append(self._output_placeholder)
                child.weights.append(None)

    def __new__(cls,
                *children: Union[Tensor, ndarray, List[Number], Number],
                name='Function'):

        if children in cls._children_history:
            cls._cache_hit = True
            return cls._children_history[children]
        else:
            cls._cache_hit = False
            creator = super(Function, cls).__new__(cls)
            cls._children_history[children] = creator
            return creator

    def __repr__(self):
        return super(Function, self).__repr__() + f'#{self.id}'

    def _generate_tensor_name(self) -> str:
        return 'Z' + self.__repr__()

    @abstractmethod
    def forward(self) -> ndarray:
        pass

    @abstractmethod
    def backward(self) -> Tuple[ndarray, ...]:
        pass

    def __call__(self) -> Tensor:
        ''' Executes forward pass and
        updates the weights (derivatives) for the dependent children.
        '''

        self._output_placeholder.value = self.forward()

        if self._output_placeholder.diff:  # Is gradient dependecy?
            self._output_placeholder.zero_grad()

            # Update the weights
            for tensor, derivative in zip(self.children, self.backward()):
                idx = next(i for i, v in enumerate(tensor.parents_outputs)
                           if (v == self._output_placeholder).all())

                tensor.weights[idx] = derivative

        return self._output_placeholder
