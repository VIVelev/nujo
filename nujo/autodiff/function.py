from abc import abstractmethod


class Function:
    ''' Abstract Base Class for functions '''

    def __init__(self, *tensors, name='undefined'):
        self.tensors = tensors
        self.name = name

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def backward(self, grad):
        pass

    def __call__(self):
        z = self.forward()

        for tensor, derivative in zip(self.tensors, self.backward(1)):
            tensor.dependencies.append(( derivative, z ))

        return z
