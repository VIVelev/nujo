import pytest
from numpy import allclose, mean, prod, sum

import nujo.math.aggregate as aggregate
from nujo.init.random import rand


def test_sum(inputs):
    assert (aggregate.sum(inputs[0]) == sum(inputs[0].value)).all()
    assert (aggregate.sum(*inputs) == sum(inputs)).all()


def test_prod(inputs):
    assert (aggregate.prod(inputs[0]) == prod(inputs[0].value)).all()
    assert (aggregate.prod(*inputs) == prod(inputs)).all()


def test_mean(inputs):
    assert allclose(aggregate.mean(inputs[0]).value, mean(inputs[0].value))
    assert (aggregate.mean(*inputs) == mean(inputs)).all()


@pytest.fixture
def inputs():
    return [
        rand(3, 3, diff=False, name='test_input_1'),
        rand(3, 3, diff=False, name='test_input_2'),
        rand(3, 3, diff=False, name='test_input_3')
    ]
