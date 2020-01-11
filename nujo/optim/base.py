__all__ = [
    'Optimizer',
]


class Optimizer:
    def __init__(self, params, lr):
        self.params = params
        self.lr = lr

    def step(self):
        pass

    def zero_grad(self):
        for l in range(len(self.params)):
            for i in range(len(self.params[l])):
                self.params[l][i].zero_grad()
