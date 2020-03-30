from copy import deepcopy

from nujo.autodiff.tensor import Tensor


class FlowSetup(type):
    ''' Flow's metaclass used to setup the computational flow
    '''
    def __call__(cls, *args, **kwargs):
        ''' Flow() init call '''
        obj = type.__call__(cls, *args, **kwargs)
        obj._register_parameters()

        return obj


class Flow(metaclass=FlowSetup):
    ''' A computational Flow

    Flow of tensors through nujo functions.

    Parameters:
    -----------
    name : string
    subflows : list of flows, only if the current flow is a supflow

    '''
    def __init__(self, name='Flow', subflows=[]):
        self.name = name
        self.is_supflow = True if len(subflows) >= 2 else False

        self.subflows = []
        self.parameters = []

        if self.is_supflow:
            self.append(*subflows)

    def _register_parameters(self) -> None:
        ''' Called after Flow.__init__ '''
        for prop_name in dir(self):
            prop = getattr(self, prop_name)
            if isinstance(prop, Tensor) and prop.diff:
                self.parameters.append(prop)

    def _generate_supflow_name(self) -> str:
        return ' >> '.join(map(lambda x: x.name, self.subflows))

    def append(self, *flows: 'Flow') -> 'Flow':
        if not self.is_supflow:
            self.subflows.append(deepcopy(self))
            self.is_supflow = True

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
        retflow = self.subflows.pop(idx)

        if len(self.subflows) == 1:
            self.__dict__ = deepcopy(self.subflows[-1].__dict__)
        else:
            self.name = self._generate_supflow_name()

        return retflow

    def forward(self, x: Tensor) -> Tensor:
        output_x = x
        for subflow in self:
            output_x = subflow(output_x)

        return output_x

    def __call__(self, *args, **kwargs) -> Tensor:
        return self.forward(*args, **kwargs)

    def __rshift__(self, other: 'Flow') -> 'Flow':
        self_subflows = self.subflows if self.is_supflow else [self]
        other_subflows = other.subflows if other.is_supflow else [other]

        return Flow(subflows=[*self_subflows, *other_subflows])

    def __getitem__(self, key: int or str) -> 'Flow':
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
