from numpy import array

from . import modes

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


    def __init__(self, value, name='undefined', children=[]):
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

        Expression.z_counter = 0

    def __add__(self, other):
        if not isinstance(other, Expression):
            other = Constant(other)

        if not modes.DIFF_ENABLED:
            return Variable(self.value + other.value, name='Add::NO_DIFF')

        z = Variable(self.value + other.value,
                        name=f'Add::Z_{Expression.get_z_count()}',
                        children=[self, other])

        self.dependencies.append(( array(1), z ))
        other.dependencies.append(( array(1), z ))

        return z

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if not isinstance(other, Expression):
            other = Constant(other)

        if not modes.DIFF_ENABLED:
            return Variable(self.value - other.value, name='Sub::NO_DIFF')

        z = Variable(self.value - other.value,
                        name=f'Sub::Z_{Expression.get_z_count()}',
                        children=[self, other])
        
        self.dependencies.append(( array(1), z ))
        other.dependencies.append(( array(-1), z ))

        return z

    def __rsub__(self, other):
        if not isinstance(other, Expression):
            other = Constant(other)
        
        return other.__sub__(self)

    def __mul__(self, other):
        if not isinstance(other, Expression):
            other = Constant(other)

        if not modes.DIFF_ENABLED:
            return Variable(self.value * other.value, name='Mul::NO_DIFF')

        z = Variable(self.value * other.value,
                        name=f'Mul::Z_{Expression.get_z_count()}',
                        children=[self, other])

        self.dependencies.append(( array(other.value), z ))
        other.dependencies.append(( array(self.value), z ))

        return z

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if not isinstance(other, Expression):
            other = Constant(other)

        if not modes.DIFF_ENABLED:
            return Variable(self.value / other.value, 'TrueDiv::NO_DIFF')

        z = Variable(self.value / other.value,
                        name=f'TrueDiv::Z_{Expression.get_z_count()}',
                        children=[self, other])

        self.dependencies.append(( array(1/other.value), z ))
        other.dependencies.append(( array((-self.value)/(other.value**2)), z ))

        return z

    def __rtruediv__(self, other):
        if not isinstance(other, Expression):
            other = Constant(other)
        
        return other.__truediv__(self)

    def __pow__(self, other):
        if not isinstance(other, Expression):
            other = Constant(other)

        if not modes.DIFF_ENABLED:
            return Variable(self.value ** other.value, name='Pow::NO_DIFF')

        z = Variable(self.value ** other.value,
                        name=f'Pow::Z_{Expression.get_z_count()}',
                        children=[self, other])

        self.dependencies.append(( array(other.value*self.value**(other.value-1)), z ))
        other.dependencies.append(( array(1), z ))

        return z

    def __matmul__(self, other):
        if not isinstance(other, Expression):
            other = Constant(other)

        if not modes.DIFF_ENABLED:
            return Variable(self.value @ other.value, name='MatMul::NO_DIFF')

        z = Variable(self.value @ other.value,
                        name=f'MatMul::Z_{Expression.get_z_count()}',
                        children=[self, other])

        self.dependencies.append(( array(other.value), z ))
        other.dependencies.append(( array(self.value), z ))

        return z

    def __rmatmul__(self, other):
        if not isinstance(other, Expression):
            other = Constant(other)

        return other.__matmul__(self)

    def __repr__(self):
        return self.name

# ====================================================================================================

class Variable(Expression):

    def __init__(self, value, name='undefined', children=[]):
        super(Variable, self).__init__(value, name, children)

    @property
    def grad(self):
        if self._grad is None:
            if len(self.dependencies) == 0:
                self._grad = array(1)
            else:
                self._grad = 0

                for weight, z in self.dependencies:
                    if not weight.shape or not z.grad.shape:
                        self._grad += weight * z.grad
                    else:
                        self._grad += weight.T * z.grad.T
        
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
        super(Constant, self).__init__(value, name=f'Const::({value})')

    @property
    def grad(self):
        if self._grad is None:
            self._grad = array(1)
        
        return self._grad

# ====================================================================================================
