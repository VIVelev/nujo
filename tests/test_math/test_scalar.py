import pytest
from numpy import (abs, allclose, ceil, exp, floor, log, log2, log10, round,
                   sqrt)

import nujo.math.scalar as scalar
from nujo.init.random import rand

# ====================================================================================================
# Test Logarithms with different bases


def test_log(inputs):
    assert (scalar.log(inputs) == log(inputs.value)).all()


def test_log2(inputs):
    assert allclose(scalar.log2(inputs).value, log2(inputs.value))


def test_log10(inputs):
    assert allclose(scalar.log10(inputs).value, log10(inputs.value))


# ====================================================================================================
# Test Exponentiation, Square Root and Absolute functions


def test_exp(inputs):
    assert allclose(scalar.exp(inputs).value, exp(inputs.value))


def test_sqrt(inputs):
    assert (scalar.sqrt(inputs) == sqrt(inputs.value)).all()


def test_abs(inputs):
    assert (scalar.abs(inputs) == abs(inputs.value)).all()


# ====================================================================================================
# Test Round, Ceil, Floor


def test_round(inputs):
    assert (scalar.round(inputs) == round(inputs.value)).all()


def test_ceil(inputs):
    assert (scalar.ceil(inputs) == ceil(inputs.value)).all()


def test_floor(inputs):
    assert (scalar.floor(inputs) == floor(inputs.value)).all()


# ====================================================================================================
# Unit Test fixtures


@pytest.fixture
def inputs():
    return rand(3, 3)


# ====================================================================================================
