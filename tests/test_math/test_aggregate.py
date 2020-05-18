import pytest
from numpy import allclose, mean, prod, sum

import nujo.math.aggregate as aggregate
from nujo.init.random import rand

# ====================================================================================================
# Test Summation


def test_sum(inputs):
    # Test Forward pass (Inner sum)
    output = aggregate.sum(inputs[0])
    assert (output == sum(inputs[0].value)).all()

    # Test Backward pass (Inner sum)
    output.backward()
    assert (inputs[0].grad.shape == inputs[0].shape)
    assert (inputs[0].grad == 1).all()

    # Test several tensors sum
    assert (aggregate.sum(*inputs) == sum(inputs)).all()


# ====================================================================================================
# Test Product


def test_prod(inputs):
    # Test Forward pass (Inner product)
    output = aggregate.prod(inputs[0])
    assert (output == prod(inputs[0].value)).all()

    # Test Backward pass (Inner product)
    output.backward()
    assert (inputs[0].grad.shape == inputs[0].shape)
    assert allclose(inputs[0].grad.value, (output / inputs[0]).value)

    # Test several tensors prod
    assert (aggregate.prod(*inputs) == prod(inputs)).all()


# ====================================================================================================
# Test Mean estimation


def test_mean(inputs):
    # Test Forward pass (Inner mean)
    output = aggregate.mean(inputs[0])
    assert allclose(output.value, mean(inputs[0].value))

    # Test Backward pass (Inner mean)
    output.backward()
    assert (inputs[0].grad.shape == inputs[0].shape)
    assert (inputs[0].grad == 1 / prod(inputs[0].shape)).all()

    # Test several tensors mean
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
