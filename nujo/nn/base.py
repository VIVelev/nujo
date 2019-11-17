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

class Flow(Transformation):

    def __init__(self):
        super(Flow, self).__init__(name='')

        self._transformations = []

    def __repr__(self):
        return self.name + ' >>'

    def append_transformations(self, *args):
        for t in args:
            self.name += ' >> ' + t.name
            self._transformations.append(t)

    def __rshift__(self, other):
        self.append_transformations(other)
        return self
