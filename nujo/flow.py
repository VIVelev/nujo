from nujo.autodiff.tensor import Tensor


class FlowSetup(type):
    ''' Flow's metaclass used to setup the flow
    '''
    def __call__(cls, *args, **kwargs):
        ''' Called after Flow.__init__ '''
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
        self.is_supflow = True if len(subflows) > 0 else False

        self.parameters = []

        if self.is_supflow:
            self.subflows = []
            self.append(*subflows)
            self.name = ' >> '.join(map(lambda x: x.name, self.subflows))

    def _register_parameters(self):
        ''' Called after Flow.__init__ '''
        for prop_name in dir(self):
            prop = getattr(self, prop_name)
            if isinstance(prop, Tensor):
                self.parameters.append(prop)

    def append(self, *flows):
        for flow in flows:
            self.subflows.append(flow)

            if getattr(flow, 'parameters', False):
                if flow.is_supflow:
                    for params in flow.parameters:
                        self.parameters.append(params)
                else:
                    self.parameters.append(flow.parameters)

        return self

    def forward(self, x):
        output_x = x
        for subflow in self:
            output_x = subflow.forward(output_x)

        return output_x

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def __rshift__(self, other):
        return Flow(subflows=[self, other])

    def __getitem__(self, key):
        return self.subflows[key]

    def __iter__(self):
        return iter(self.subflows)

    def __repr__(self):
        return '<|' + self.name + '>'
