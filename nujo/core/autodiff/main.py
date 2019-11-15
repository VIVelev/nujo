import numpy as np


# ====================================================================================================
# ====================================================================================================

class Expression:

    def __init__(self, value, children=[]):
        self.value = value
        self.children = children

        self._dependencies = []
        self._grad = None

    @property
    def dependencies(self):
        return self._dependencies

    @property
    def grad(self):
        pass

    @grad.setter
    def grad(self, value):
        self._grad = value

    def backward(self, prev_grad=1):
        for child in self.children:
            child.backward(prev_grad=self.grad)

    def __add__(self, other):
        z = Variable(self.value + other.value,
                        children=[self, other])

        self.dependencies.append((1, z))
        other.dependencies.append((1, z))

        return z

    def __mul__(self, other):
        z = Variable(self.value * other.value,
                        children=[self, other])

        self.dependencies.append((other.value, z))
        other.dependencies.append((self.value, z))

        return z

# ====================================================================================================

class Variable(Expression):

    def __init__(self, value, children=[]):
        super(Variable, self).__init__(value, children)

    @property
    def grad(self):
        if self._grad is None:
            if len(self.dependencies) == 0:
                self._grad = 1
            else:
                self._grad = np.sum(weight * z.grad for weight, z in self.dependencies)
        
        return self._grad

    def zero_grad(self):
        self.grad = np.zeros_like(self.value)

# ====================================================================================================

class Constant(Expression):

    def __init__(self, value):
        super(Constant, self).__init__(value)

    @property
    def grad(self):
        if self._grad is None:
            self._grad = 1
        
        return self._grad

# ====================================================================================================
