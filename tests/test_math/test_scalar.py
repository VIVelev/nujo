import pytest
from numpy import abs, ceil, exp, floor, isclose, log, log2, log10, round, sqrt

import nujo.math.scalar as scalar
from nujo.init.random import rand

# ====================================================================================================
# Test Logarithms with different bases


def test_log(inputs):
    target = log(inputs.value)

    assert (scalar.log(inputs) == target).all()
    assert (scalar.log(inputs.value) == target).all()


def test_log2(inputs):
    target = log2(inputs.value)

    assert isclose(scalar.log2(inputs).value, target).all()
    assert isclose(scalar.log2(inputs.value).value, target).all()


def test_log10(inputs):
    target = log10(inputs.value)

    assert isclose(scalar.log10(inputs).value, target).all()
    assert isclose(scalar.log10(inputs.value).value, target).all()


# ====================================================================================================
# Test Exponentiation, Square Root and Absolute functions


def test_exp(inputs):
    target = exp(inputs.value)

    assert isclose(scalar.exp(inputs).value, target).all()
    assert isclose(scalar.exp(inputs.value).value, target).all()


def test_sqrt(inputs):
    target = sqrt(inputs.value)

    assert (scalar.sqrt(inputs) == target).all()
    assert (scalar.sqrt(inputs.value) == target).all()


def test_abs(inputs):
    target = abs(inputs.value)

    assert (scalar.abs(inputs) == target).all()
    assert (scalar.abs(inputs.value) == target).all()


# ====================================================================================================
# Test Round, Ceil, Floor


def test_round(inputs):
    target = round(inputs.value)

    assert (scalar.round(inputs) == target).all()
    assert (scalar.round(inputs.value) == target).all()


def test_ceil(inputs):
    target = ceil(inputs.value)

    assert (scalar.ceil(inputs) == target).all()
    assert (scalar.ceil(inputs.value) == target).all()


def test_floor(inputs):
    target = floor(inputs.value)

    assert (scalar.floor(inputs) == target).all()
    assert (scalar.floor(inputs.value) == target).all()


# ====================================================================================================
# Unit Test fixtures


@pytest.fixture
def inputs():
    return rand(3, 3)


# ====================================================================================================
