import pytest

import nujo.math.scalar as scalar
from nujo.init.random import rand


def test_log(input):
    scalar.log(input)
    scalar.log(input.value)
    assert True


def test_log2(input):
    scalar.log2(input)
    scalar.log2(input.value)
    assert True


def test_log10(input):
    scalar.log10(input)
    scalar.log10(input.value)
    assert True


def test_exp(input):
    scalar.exp(input)
    scalar.exp(input.value)
    assert True


def test_sqrt(input):
    scalar.sqrt(input)
    scalar.sqrt(input.value)
    assert True


def test_abs(input):
    scalar.abs(input)
    scalar.abs(input.value)
    assert True


def test_round(input):
    scalar.round(input)
    scalar.round(input.value)
    assert True


def test_ceil(input):
    scalar.ceil(input)
    scalar.ceil(input.value)
    assert True


def test_floor(input):
    scalar.floor(input)
    scalar.floor(input.value)
    assert True


@pytest.fixture
def input():
    return rand(3, 3, diff=False, name='test_input')
