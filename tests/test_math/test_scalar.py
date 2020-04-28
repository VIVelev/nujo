import pytest
from numpy import abs, ceil, exp, floor, isclose, log, log2, log10, round, sqrt

import nujo.math.scalar as scalar
from nujo.init.random import rand


def test_log(input):
    target = log(input.value)

    assert (scalar.log(input) == target).all()
    assert (scalar.log(input.value) == target).all()


def test_log2(input):
    target = log2(input.value)

    assert isclose(scalar.log2(input).value, target).all()
    assert isclose(scalar.log2(input.value).value, target).all()


def test_log10(input):
    target = log10(input.value)

    assert isclose(scalar.log10(input).value, target).all()
    assert isclose(scalar.log10(input.value).value, target).all()


def test_exp(input):
    target = exp(input.value)

    assert isclose(scalar.exp(input).value, target).all()
    assert isclose(scalar.exp(input.value).value, target).all()


def test_sqrt(input):
    target = sqrt(input.value)

    assert (scalar.sqrt(input) == target).all()
    assert (scalar.sqrt(input.value) == target).all()


def test_abs(input):
    target = abs(input.value)

    assert (scalar.abs(input) == target).all()
    assert (scalar.abs(input.value) == target).all()


def test_round(input):
    target = round(input.value)

    assert (scalar.round(input) == target).all()
    assert (scalar.round(input.value) == target).all()


def test_ceil(input):
    target = ceil(input.value)

    assert (scalar.ceil(input) == target).all()
    assert (scalar.ceil(input.value) == target).all()


def test_floor(input):
    target = floor(input.value)

    assert (scalar.floor(input) == target).all()
    assert (scalar.floor(input.value) == target).all()


@pytest.fixture
def input():
    return rand(3, 3, diff=False, name='test_input')
