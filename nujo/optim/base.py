__all__ = [
    'Optimizer',
]


class Optimizer:
    def __init__(self, net, lr):
        self.net = net
        self.lr = lr

    def step(self):
        pass
