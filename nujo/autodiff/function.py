from abc import abstractmethod
from numbers import Number
from typing import Dict, List, Tuple, Union

from numpy import ndarray

from nujo.autodiff import modes
from nujo.autodiff._node import _Node
from nujo.autodiff.tensor import Tensor


class Function(_Node, object):
    ''' Base Class for functions

    Functions are applied to tensors. They take multiple
    tensors as input and produces only one tensor as output.
    They do NOT change tensors in-place.

    Functions were also written so they reuse the input/output tensors
    when possible, which results in the computation graph being:
     - "Dynamically defined, statically evaluated."
    taking the best from both worlds.

    Parameters:
    -----------
     - children : varargs, the inpute tensors
     - name : string, the name of the function

    '''

    _children_history: Dict[str, 'Function'] = {}
    ''' Cache where input tensors for
    the current function type are stored.
    '''

    _cache_hit = False
    ''' Flag signaling cache hit/miss.
    '''
    def __init__(self,
                 *children: Union[Tensor, ndarray, List[Number], Number],
                 name='Function'):

        if self._cache_hit:
            return

        super(Function, self).__init__(*children, name=name)

        # This output placeholder is reused when possible
        self._output_placeholder = Tensor(
            None,
            diff=any([x.diff for x in self.children]) and modes.DIFF_ENABLED,
            creator=self if modes.DIFF_ENABLED else None,
            name=self._generate_tensor_name())

        if modes.DIFF_ENABLED:  # If graph building is enabled.
            # Allocate space for parent's output (output placeholder)
            # and its weight (derivative).

            for child in self.children:
                child.parents_outputs.append(self._output_placeholder)
                child.weights.append(None)

    def __new__(cls, *children: Union[Tensor, ndarray, List[Number], Number],
                **kwargs):
        ''' Used to review the cache for hit and return the cached tensor
        or otherwise add the new tensor to the cache.
        '''

        if modes.DIFF_ENABLED:
            key = str(hash(cls))  # Inlcude the function type hash in the key
            # Include the arguments' uids in the key
            key += ''.join((str(x.id) if isinstance(x, Tensor) else str(x)
                            for x in children))

            if key in cls._children_history:
                cls._cache_hit = True
                return cls._children_history[key]
            else:
                cls._cache_hit = False
                creator = super(Function, cls).__new__(cls)
                cls._children_history[key] = creator
                return creator

        else:
            cls._cache_hit = False
            return super(Function, cls).__new__(cls)

    def __repr__(self):
        return super(Function, self).__repr__() + f'#{self.id}'

    def _generate_tensor_name(self) -> str:
        return 'Z' + self.__repr__()

    @abstractmethod
    def forward(self) -> ndarray:
        ''' Implement forward pass of the function here.
        '''

        pass

    @abstractmethod
    def backward(self) -> Tuple[ndarray, ...]:
        ''' Implement backward pass of the function here
        (a.k.a. derivative calculation).
        '''

        pass

    def __call__(self) -> Tensor:
        ''' Executes forward pass and
        updates the weights (derivatives) for the dependent children.
        '''

        # Forward pass
        self._output_placeholder.value = self.forward()

        if self._output_placeholder.diff:  # Is gradient dependecy?
            self._output_placeholder.zero_grad()

            # Update the weights
            for tensor, derivative in zip(self.children, self.backward()):
                idx = next(i for i, v in enumerate(tensor.parents_outputs)
                           if v is self._output_placeholder)

                tensor.weights[idx] = derivative

        return self._output_placeholder
