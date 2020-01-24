__all__ = [
    'Transformation',
    'Flow',
]


class Transformation:
    def __init__(self, name='undefined'):
        self.name = name

    def __repr__(self):
        return self.name

    def __rshift__(self, other):
        flow = Flow()
        flow.append_transformations(self, other)

        return flow

    def __call__(self, input):
        return self.forward(input)

    def forward(self, input):
        pass


class Flow(Transformation):
    def __init__(self):
        super(Flow, self).__init__(name='')

        self._transformations = []
        self.parameters = []

    def __repr__(self):
        return self.name + ' >>'

    def __rshift__(self, other):
        self.append_transformations(other)
        return self

    def __call__(self, input):
        output = input
        for t in self._transformations:
            output = t(output)

        return output

    def __getitem__(self, key):
        return self._transformations[key]

    def __iter__(self):
        return iter(self._transformations)

    def append_transformations(self, *args):
        for t in args:
            self.name += ' >> ' + t.name
            self._transformations.append(t)

            if 'parameters' in dir(t):
                self.parameters.append(t.parameters)
