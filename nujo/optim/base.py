__all__ = [
    'Optimizer',
]


class Optimizer:
    def __init__(self, parameters, lr):
        self.parameters = parameters
        self.lr = lr

    def step(self):
        pass

    def zero_grad(self):
        for l in range(len(self.parameters)):
            for i in range(len(self.parameters[l])):
                self.parameters[l][i].zero_grad()
