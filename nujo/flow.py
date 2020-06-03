''' a chainable computation Flow
'''

from abc import abstractmethod
from copy import deepcopy
from itertools import chain
from typing import List, Union

from nujo.autodiff.tensor import Tensor


class _FlowMeta(type):
    ''' Flow's metaclass used to setup the computation flow
    '''
    def __call__(cls, *args, **kwargs):
        ''' Flow's __init__ '''
        obj = type.__call__(cls, *args, **kwargs)  # Call __init__

        if len(obj) == 0:  # If no chain has been setup
            obj._register_parameters()
            # Set the chain, starting with the current flow
            obj = Flow(_chain=[obj])

        return obj


class Flow(metaclass=_FlowMeta):
    ''' A chainable computation Flow

    A Flow is just a sequance of functions (addition, multiplication, etc.)
    that are grouped in a single object (Flow) and can be applied on a tensor.

    Each nujo Flow has a list of flow objects (a chain) that a tensor will pass
    through when the Flow is called on that tensor.

    This allows the chaining of flows (connecting two or more chains together).

    Parameters:
    -----------
     - name : string, idetifier of the current flow

    '''
    def __init__(self, name='Flow', _chain: List['Flow'] = []):
        self.name = name
        self._chain = _chain

        if len(self._chain):  # If there is a chain
            self.name = self._generate_chain_name()

    # setup methods

    def _register_parameters(self) -> None:
        ''' Tensor parameters registration - called after Flow.__init__

        Makes all tensors bounded to `self` diff enabled (sets their `diff`
        to `True`).

        Called only once, when the chain for the current flow is being created.

        '''

        for prop_name in dir(self):
            prop = getattr(self, prop_name)

            if isinstance(prop, Tensor):
                prop.diff = True

    def _generate_chain_name(self) -> str:
        return ' >> '.join(map(lambda x: x.name, self._chain))

    # parameters generators

    def parameters(self) -> Tensor:
        ''' Generator for all the parameters of the current flow
        '''

        for param in self._total_parameters():
            yield param

    def _total_parameters(self) -> Tensor:
        ''' Returns an iterable of all the parameters of the current flow

        Including those of other flows that are used in the current one
        (namely other flows bounded to `self`).

        '''

        total_params = [self._current_parameters()]

        for prop_name in dir(self):
            prop = getattr(self, prop_name)

            if isinstance(prop, Flow):
                total_params.append(prop.parameters())

        return chain(*total_params)

    def _current_parameters(self) -> Tensor:
        ''' Generator for the current tensor parameters bounded to `self`
        '''

        for flow in self._chain:
            for prop_name in dir(flow):
                prop = getattr(flow, prop_name)

                if isinstance(prop, Tensor):
                    yield prop

    # API methods

    def append(self, *flows: 'Flow') -> 'Flow':
        ''' Flow Append

        Connect the current chain with those of `flows` by adding them
        at the end.

        Parameters:
        -----------
         - flows : varargs, the flows to append, sequantially

        Returns:
        --------
         - flow : Flow, the total computation flow

        '''

        for flow in flows:
            for chain_section in flow:  # Iterate over the chain
                # Connect with the current chain
                self._chain.append(chain_section)

        self.name = self._generate_chain_name()  # Update the chain name
        return self

    def pop(self, idx=-1) -> 'Flow':
        ''' Flow Pop

        Removes a flow (and it's chain) at a given index, defaults to
        the last one (-1).

        Parameters:
        -----------
         - idx : integer, index of the flow to remove

        Returns:
        --------
         - flow : Flow, the total computation flow

        '''

        retflow = self._chain.pop(idx)
        self.name = self._generate_chain_name()

        return retflow

    def copy(self) -> 'Flow':
        ''' Make a copy of the flow
        '''

        return deepcopy(self)

    @abstractmethod
    def forward(self, *args, **kwargs) -> Tensor:
        ''' Flow Forward

        The flow computation is defined here.

        '''

        pass

    # methods implementing the flow functionality

    def __call__(self, *args, **kwargs) -> Tensor:
        output = self[0].forward(*args, **kwargs)

        for flow in self[1:]:
            output = flow.forward(output, **kwargs)

        return output

    def __rshift__(self, other: 'Flow') -> 'Flow':
        ''' Chaining operator

        Example:
            >>> a = nj.Flow()
            >>> b = nj.Flow()
            >>> chained_flow = a >> b
            >>> result = chained_flow(...)
            >>> ...

        '''

        return Flow(_chain=[*list(self), *list(other)])

    def __getitem__(self, key: Union[int, str]) -> 'Flow':
        '''Access flows in the chain by index/name

        Example:
            >>> a = nj.Flow('A')
            >>> b = nj.Flow('B')
            >>> chained_flow = a >> b
            >>> chained_flow[0]  # a flow (chain section) can be get by index
            'A' (this is the repr for `a`)
            >>> chained_flow['A']  # can also be get by name
            'A'

        '''

        if type(key) is str:
            flow = next((x for x in self._chain if x.name == key), None)
            if flow is not None:
                return flow
            else:
                raise ValueError(f'Could not find a flow named: {key}')
        else:
            return self._chain[key]

    def __iter__(self):
        return iter(self._chain)

    def __len__(self):
        return len(self._chain)

    def __repr__(self):
        return '<|' + self.name + '>'
