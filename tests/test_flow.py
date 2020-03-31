import pytest

from nujo.flow import Flow


def test_chaining(flows):
    _, _, supflow = flows

    assert supflow.is_supflow
    assert supflow.name == 'mul2 >> add1'
    assert len(supflow.subflows) == 2


def test_forward(flows):
    mul2, _, supflow = flows

    assert mul2(42) == 42 * 2
    assert supflow(42) == 42 * 2 + 1


def test_append(flows):
    mul2, add1, supflow = flows

    assert not mul2.is_supflow
    mul2_add1 = mul2.append(add1)
    assert mul2_add1.is_supflow

    assert mul2_add1.name == 'mul2 >> add1'
    assert mul2_add1(42) == 42 * 2 + 1

    assert supflow.is_supflow
    supflow = supflow.append(mul2)
    assert supflow.is_supflow

    assert supflow.name == 'mul2 >> add1 >> mul2'
    assert supflow(42) == (42 * 2 + 1) * 2


def test_pop(flows):
    mul2, add1, supflow = flows

    poped = supflow.pop()
    assert poped is add1
    assert supflow.is_supflow

    assert supflow.name is mul2.name
    assert supflow(42) == mul2(42) == 42 * 2


@pytest.fixture
def flows():
    class Mul2(Flow):
        def forward(self, x):
            return x * 2

    class Add1(Flow):
        def forward(self, x):
            return x + 1

    mul2 = Mul2('mul2')
    add1 = Add1('add1')
    supflow = mul2 >> add1

    return mul2, add1, supflow
