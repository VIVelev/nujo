from nujo.autodiff.function import Function

__all__ = [
    'Add',
]


class Addition(Function):

    @staticmethod
    def forward(input_a, input_b):
        return input_a + input_b

    @staticmethod
    def backward(input):
        return input

class Multiplication(Function):

    @staticmethod
    def forward(input_a, input_b):
        pass
