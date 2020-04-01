''' a computational Flow
'''

from copy import deepcopy

from nujo.autodiff.tensor import Tensor


class _FlowSetup(type):
    ''' Flow's metaclass used to setup the computational flow
    '''
    def __call__(cls, *args, **kwargs):
        ''' Flow() init call '''
        obj = type.__call__(cls, *args, **kwargs)
        obj._register_parameters()

        return obj


class Flow(metaclass=_FlowSetup):
    ''' A computational Flow

    Flow of tensors through nujo functions.

    Parameters:
    -----------
    name : string
    subflows : list of flows, only if the current flow is a supflow

    '''
    def __init__(self, name='Flow', subflows=[]):
        self.name = name
        self.is_supflow = True if subflows else False

        self.subflows = []
        self.parameters = []

        if self.is_supflow:
            self.append(*subflows)

    def _register_parameters(self) -> None:
        ''' Called after Flow.__init__
        '''

        for prop_name in dir(self):
            prop = getattr(self, prop_name)
            if isinstance(prop, Tensor) and prop.diff:
                self.parameters.append(prop)

    def _generate_supflow_name(self) -> str:
        return ' >> '.join(map(lambda x: x.name, self.subflows))

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

        if not self.is_supflow:
            flows = list(flows)
            flows.insert(0, self)
            return Flow(subflows=flows)

        for flow in flows:
            self.subflows.append(flow)

            if flow.parameters:
                if flow.is_supflow:
                    for params in flow.parameters:
                        self.parameters.append(params)
                else:
                    self.parameters.append(flow.parameters)

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

    def forward(self, x: Tensor) -> Tensor:
        ''' Flow Forward

        The flow computation is defined here.

        Parameters:
        -----------
        x : Tensor, input tensor

        Returns:
        --------
        res : Tensor, computed result

        '''

        output_x = x
        for subflow in self:
            output_x = subflow(output_x)

        return output_x

    def copy(self):
        ''' Make a copy of the current flow
        '''

        return deepcopy(self)

    def __call__(self, *args, **kwargs) -> Tensor:
        return self.forward(*args, **kwargs)

    def __rshift__(self, other: 'Flow') -> 'Flow':
        ''' Chaining operator

        Example:
            >>> a = nj.Flow()
            >>> b = nj.Flow()
            >>> chained_supflow = a >> b
            >>> result = chained_supflow(...)
            >>> ...

        '''

        self_subflows = self.subflows if self.is_supflow else [self]
        other_subflows = other.subflows if other.is_supflow else [other]

        return Flow(subflows=[*self_subflows, *other_subflows])

    def __getitem__(self, key: int or str) -> 'Flow':
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

    def __repr__(self):
        return '<|' + self.name + '>'
