from abc import abstractmethod
from numbers import Number
from typing import Any, Dict, Iterable, List, TypeVar, Union

from numpy import ndarray

import nujo.autodiff.modes as modes
from nujo.autodiff._node import _Node
from nujo.autodiff.tensor import Tensor

# ====================================================================================================


class _FunctionMeta(type):
    def __call__(cls, *children: Union[Tensor, ndarray, List[Number], Number],
                 **kwargs):
        ''' Used to lookup the cache for an already defined function of
        the current type using the current `children` as inputs, and reuse
        it. If a function satisfying this requirements could not be found,
        a new function is created and added to the cache, in order to be,
        potentially, later reused.

        '''
        obj = cls.__new__(cls, *children, **kwargs)

        # Only cache functions that are in the computation graph
        if modes.DIFF_ENABLED:
            key = _get_function_identifier(cls, children)
            cache = cls._func_children_lookup_cache

            if key in cache:
                return cache[key]

            else:
                cls.__init__(obj, *children, **kwargs)
                cache[key] = obj
                return obj

        # Otherwise - standard call
        cls.__init__(obj, *children, **kwargs)
        return obj


# ====================================================================================================


class Function(_Node, metaclass=_FunctionMeta):
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

    '''

    _func_children_lookup_cache: Dict[str, 'Function'] = {}
    ''' Cache used to lookup for functions that may have already been defined
    in the computation graph.

     - key : hash(FuncType) + (children's identifiers);
     use `_get_function_identifier` to obtain a key
     - value : the already defined function which can be reused

    '''

    T = TypeVar('T', Tensor, ndarray)

    def __init__(self, *children: Union[Tensor, ndarray, List[Number],
                                        Number]):

        super(Function, self).__init__(*_parse_inputs(children),
                                       name=self.__class__.__name__)

        # This output placeholder is reused when possible
        self._output_placeholder = Tensor(
            None,
            diff=any(x.diff for x in self.children) and modes.DIFF_ENABLED,
            creator=self if modes.DIFF_ENABLED else None,
            name=self._generate_tensor_name())

        if modes.DIFF_ENABLED:  # If graph building is enabled.
            # Allocate space for parent's output (output placeholder)
            for child in self.children:
                child.parents_outputs.append(self._output_placeholder)

    def __repr__(self):
        return super(Function, self).__repr__() + f'#{self.id}'

    def _generate_tensor_name(self) -> str:
        return 'Z' + self.__repr__()

    @abstractmethod
    def forward(self) -> ndarray:
        ''' Implement forward pass of the function here.

        Use the `self.children` list to access the inputs.

        '''

        pass

    @abstractmethod
    def backward(self, idx: int, accum_grad: T) -> T:
        ''' Implement backward pass of the function here

        Compute the gradient of children[idx] w.r.t. output of the
        computation graph from the accumulated gradient (the gradient
        of the output of the function w.r.t. the output of the graph).

        Parameters:
        -----------
        - idx : int, the index of the children for which to compute the
         gradient w.r.t. output of the computation graph
        - accum_grad : T (Tensor or ndarray), the accumulated grad in the graph
         so far, you can otherwise think of it as the gradient of the output of
         the function w.r.t. the output of the graph.

            - `accum_grad` is Tensor if differentiantion is enabled
             (`DIFF_ENABLED`) and the children has opted for differentiation
             (`diff` is True), thus the computations will be recorded in the
             computation graph and higher-order derivatives could be computed.
            - otherwise, `accum_grad` is ndarray and the computations are not
             recorded; ndarrays are used since the computations with them are
             more efficient.

        Returns:
        --------
        - grad : T (Tensor or ndarray), the computed gradient of
         `self.children[idx]`

        '''

        pass

    def __call__(self) -> Tensor:
        ''' Executes cached forward pass
        '''

        # Forward pass
        self._output_placeholder.value = self.forward()
        return self._output_placeholder


# ====================================================================================================


def _parse_inputs(inputs: Iterable[Any]) -> List[Tensor]:
    ''' Parse all inputs that are not Nodes to Tensors
    '''

    return [
        x if isinstance(x, _Node) else Tensor(x, name=str(x)) for x in inputs
    ]


# ====================================================================================================


def _get_function_identifier(func_type: type, inputs: Iterable[Any]) -> str:
    ''' Returns a string identifier for the current function type and its inputs,
    used for a key in the cache.

    '''

    key = str(hash(func_type))  # Inlcude the function type hash in the key
    # Include the inputs' (children's) identifiers in the key
    key += ''.join(('T' + str(x.id) if isinstance(x, Tensor) else 'P' + str(x)
                    for x in inputs))

    # 'T' and 'P' signatures were added in order to avoid
    # collisions between Tensor and Python values

    return key


# ====================================================================================================
