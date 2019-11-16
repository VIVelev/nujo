import numpy as np


__all__ = [
    'Expression',
    'Variable',
    'Constant',
]

# ====================================================================================================
# ====================================================================================================

class Expression:

    z_counter = 0

    @staticmethod
    def get_z_count():
        Expression.z_counter += 1
        return Expression.z_counter


    def __init__(self, value, name='', children=[]):
        self.value = value
        self.name = name
        self.children = children

        self._dependencies = []
        self._grad = None

    @property
    def dependencies(self):
        return self._dependencies

    @dependencies.setter
    def dependencies(self, value):
        self._dependencies = value

    @property
    def grad(self):
        pass

    def backward(self):
        _ = self.grad
        for child in self.children:
            child.backward()

    def __add__(self, other):
        if not isinstance(other, Expression):
            other = Constant(other)

        z = Variable(self.value + other.value,
                        name=f'z{Expression.get_z_count()}',
                        children=[self, other])

        self.dependencies.append((1, z))
        other.dependencies.append((1, z))

        return z

    def __sub__(self, other):
        if not isinstance(other, Expression):
            other = Constant(other)

        z = Variable(self.value - other.value,
                        name=f'z{Expression.get_z_count()}',
                        children=[self, other])
        
        self.dependencies.append((1, z))
        other.dependencies.append((-1, z))

        return z

    def __mul__(self, other):
        if not isinstance(other, Expression):
            other = Constant(other)

        z = Variable(self.value * other.value,
                        name=f'z{Expression.get_z_count()}',
                        children=[self, other])

        self.dependencies.append((other.value, z))
        other.dependencies.append((self.value, z))

        return z

    def __truediv__(self, other):
        if not isinstance(other, Expression):
            other = Constant(other)

        z = Variable(self.value / other.value,
                        name=f'z{Expression.get_z_count()}',
                        children=[self, other])

        self.dependencies.append((1/other.value, z))
        other.dependencies.append(((-self.value)/(other.value**2), z))

        return z

    def __pow__(self, other):
        if not isinstance(other, Expression):
            other = Constant(other)

        z = Variable(self.value**other.value,
                        name=f'z{Expression.get_z_count()}',
                        children=[self, other])

        self.dependencies.append((other.value*self.value**(other.value-1), z))
        other.dependencies.append((np.log(self.value)*self.value**other.value, z))

        return z

    def __sum__(self):
        if isinstance(self.value, list):
            return sum(self.value)
        else:
            return self.value

    def __repr__(self):
        return self.name

# ====================================================================================================

class Variable(Expression):

    def __init__(self, value, name='', children=[]):
        super(Variable, self).__init__(value, name, children)

    def __repr__(self):
        return self.name

    @property
    def grad(self):
        if self._grad is None:
            if len(self.dependencies) == 0:
                self._grad = 1
            else:
                self._grad = np.sum(weight * z.grad for weight, z in self.dependencies)
        
        return self._grad

    @grad.setter
    def grad(self, value):
        self._grad = value

    def zero_grad(self):
        self.grad = None
        self.dependencies = []

# ====================================================================================================

class Constant(Expression):

    def __init__(self, value):
        super(Constant, self).__init__(value)

    def __repr__(self):
        return f'Constant({self.value})'

    @property
    def grad(self):
        if self._grad is None:
            self._grad = 1
        
        return self._grad

# ====================================================================================================
