from nujo.autodiff.constant import Constant
from nujo.autodiff.function import Function
from nujo.autodiff.misc import generate_tensor_name
from nujo.autodiff.modes import DIFF_ENABLED
from nujo.autodiff.variable import Variable

__all__ = [
    'Addition',
]


class Addition(Function):

    def __init__(self, input_a, input_b, name='Add'):
        super(Addition, self).__init__(input_a, input_b, name=name)

    def forward(self):
        return Variable(self.inputs[0] + self.inputs[1],
            name = generate_tensor_name(
                self.inputs[0].z_counter.get(),
                self.name),
            children = self.inputs if DIFF_ENABLED else [])

    def backward(self, grad):
        return grad, grad
