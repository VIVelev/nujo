import pytest
from numpy import allclose, mean, prod, sum

import nujo.math.aggregate as aggregate
from nujo.init.random import rand

# ====================================================================================================
# Test Summation


def test_sum(inputs):
    output = aggregate.sum(inputs[0])
    assert (output == sum(inputs[0].value)).all()
    assert (inputs[0].grad == 1).all()

    assert (aggregate.sum(*inputs) == sum(inputs)).all()


# ====================================================================================================
# Test Product


def test_prod(inputs):
    output = aggregate.prod(inputs[0])
    assert (output == prod(inputs[0].value)).all()
    assert allclose(inputs[0].grad.value, (output / inputs[0]).value)

    assert (aggregate.prod(*inputs) == prod(inputs)).all()


# ====================================================================================================
# Test Mean estimation


def test_mean(inputs):
    output = aggregate.mean(inputs[0])
    assert allclose(output.value, mean(inputs[0].value))
    assert (inputs[0].grad == 1 / prod(inputs[0].shape)).all()

    assert (aggregate.mean(*inputs) == mean(inputs)).all()


# ====================================================================================================
# Unit Test fixtures


@pytest.fixture
def inputs():
    return [
        rand(3, 3, diff=True),
        rand(3, 3, diff=True),
        rand(3, 3, diff=True),
    ]


# ====================================================================================================
