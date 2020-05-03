''' a computational Flow
'''

from abc import abstractmethod
from copy import deepcopy
from typing import List, Union

from nujo.autodiff.tensor import Tensor


class _FlowSetup(type):
    ''' Flow's metaclass used to setup the computational flow
    '''
    def __call__(cls, *args, **kwargs):
        ''' Flow() init call '''
        obj = type.__call__(cls, *args, **kwargs)

        if not obj.subflows:
            obj = Flow(name=obj.name, subflows=[obj])

        obj._register_parameters()
        obj.name = obj._generate_supflow_name()
        return obj


class Flow(metaclass=_FlowSetup):
    ''' A computational Flow

    Flow of tensors through nujo functions.

    Parameters:
    -----------
    name : string
    subflows : list of flows that will run when called

    '''
    def __init__(self, name='Flow', subflows: List['Flow'] = []):
        self.name = name
        self.subflows = subflows

    def _register_parameters(self) -> None:
        ''' Called after Flow.__init__
        '''

        for flow in self.subflows:
            for prop_name in dir(flow):
                prop = getattr(flow, prop_name)

                if isinstance(prop, Tensor):
                    prop.diff = True

    def _generate_supflow_name(self) -> str:
        return ' >> '.join(map(lambda x: x.name, self.subflows))

    def parameters(self) -> Tensor:
        for flow in self.subflows:
            for prop_name in dir(flow):
                prop = getattr(flow, prop_name)

                if isinstance(prop, Tensor):
                    updated = (yield prop)
                    yield
                    if updated is not None:
                        prop <<= updated

    def append(self, *flows: 'Flow') -> 'Flow':
        ''' Flow Append

        Appends a new Flow on top of the current one.

        Parameters:
        -----------
        flows : varargs, the flows to append, sequantially

        Returns:
        --------
        supflow : Flow, the total computational flow

        '''

        for flow in flows:
            if len(flow):
                for subflow in flow:
                    self.subflows.append(subflow)
            else:
                self.subflows.append(flow)

        self.name = self._generate_supflow_name()
        return self

    def pop(self, idx=-1) -> 'Flow':
        ''' Flow Pop

        Removes a flow at a given index, defaults to the last one (-1).

        Once a supflow is a supflow it stays a supflow.
        No mather how many subflows it contains.

        Parameters:
        -----------
        idx : integer, index of the flow to remove

        Returns:
        --------
        flow : Flow, the total computational flow

        '''

        retflow = self.subflows.pop(idx)
        self.name = self._generate_supflow_name()

        return retflow

    def copy(self) -> 'Flow':
        ''' Make a copy of the current flow
        '''

        return deepcopy(self)

    @abstractmethod
    def forward(self, *args, **kwargs) -> Tensor:
        ''' Flow Forward

        The flow computation is defined here.

        '''

        pass

    def __call__(self, *args, **kwargs) -> Tensor:
        output = self[0].forward(*args, **kwargs)

        for subflow in self[1:]:
            output = subflow.forward(output, **kwargs)

        return output

    def __rshift__(self, other: 'Flow') -> 'Flow':
        ''' Chaining operator

        Example:
            >>> a = nj.Flow()
            >>> b = nj.Flow()
            >>> chained_supflow = a >> b
            >>> result = chained_supflow(...)
            >>> ...

        '''

        return Flow(subflows=[*self.subflows, *other.subflows])

    def __getitem__(self, key: Union[int, str]) -> 'Flow':
        ''' Subflow getter of a supflow

        Example:
            >>> a = nj.Flow('A')
            >>> b = nj.Flow('B')
            >>> chained_supflow = a >> b
            >>> chained_supflow[0]  # a subflow can be get by index
            'A' (this is the repr for `a`)
            >>> chained_supflow['A']  # can also be get by name
            'A'

        '''

        if type(key) is str:
            flow = next((x for x in self.subflows if x.name == key), None)
            if flow is not None:
                return flow
            else:
                raise ValueError(f'Could not find a flow named: {key}')
        else:
            return self.subflows[key]

    def __iter__(self):
        return iter(self.subflows)

    def __len__(self):
        return len(self.subflows)

    def __repr__(self):
        return '<|' + self.name + '>'
