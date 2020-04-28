import pytest
from numpy import max, mean, median, min, prod, std, sum, var

import nujo.math.aggregate as aggregate
from nujo.init.random import rand


def test_sum(input):
    assert (aggregate.sum(input[0]) == sum(input[0].value)).all()
    assert (aggregate.sum(*input) == sum([x.value for x in input])).all()


def test_prod(input):
    assert (aggregate.prod(input[0]) == prod(input[0].value)).all()
    assert (aggregate.prod(*input) == prod([x.value for x in input])).all()


def test_mean(input):
    assert (aggregate.mean(input[0]) == mean(input[0].value)).all()
    assert (aggregate.mean(*input) == mean([x.value for x in input])).all()


def test_median(input):
    assert (aggregate.median(input[0]) == median(input[0].value)).all()
    assert (aggregate.median(*input) == median([x.value for x in input])).all()


def test_min(input):
    assert (aggregate.min(input[0]) == min(input[0].value)).all()
    assert (aggregate.min(*input) == min([x.value for x in input])).all()


def test_max(input):
    assert (aggregate.max(input[0]) == max(input[0].value)).all()
    assert (aggregate.max(*input) == max([x.value for x in input])).all()


def test_stddev(input):
    assert (aggregate.stddev(input[0]) == std(input[0].value)).all()
    assert (aggregate.stddev(*input) == std([x.value for x in input])).all()


def test_variance(input):
    assert (aggregate.variance(input[0]) == var(input[0].value)).all()
    assert (aggregate.variance(*input) == var([x.value for x in input])).all()


@pytest.fixture
def input():
    return [
        rand(3, 3, diff=False, name='test_input_1'),
        rand(3, 3, diff=False, name='test_input_2'),
        rand(3, 3, diff=False, name='test_input_3')
    ]
