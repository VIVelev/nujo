import pytest

from nujo.flow import Flow

# ====================================================================================================
# Test custom Flow creation


def test_custom_flow_creation():
    class CustomFlow(Flow):
        def forward(self, x):
            return x**2 + 42

    flow = CustomFlow('SomeFlowName')

    assert flow.name == 'SomeFlowName'
    assert repr(flow) == '<|SomeFlowName>'
    assert flow[0].name == flow.name

    assert flow(9) == 9**2 + 42


# ====================================================================================================
# Test Flow append and pop


def test_append(flows):
    mul2, add1, supflow = flows

    # -------------------------

    mul2_add1 = mul2.copy().append(add1)
    assert len(mul2_add1) == 2
    assert mul2_add1[1] is add1[0]

    assert mul2_add1[0].name == 'mul2'
    assert mul2_add1[1].name == 'add1'
    assert mul2_add1.name == 'mul2 >> add1'
    assert mul2_add1(42) == 42 * 2 + 1

    # -------------------------

    assert len(supflow) == 2
    supflow = supflow.append(mul2)
    assert len(supflow) == 3
    assert supflow[2] is mul2[0]

    assert supflow[0].name == 'mul2'
    assert supflow[1].name == 'add1'
    assert supflow[2].name == 'mul2'
    assert supflow.name == 'mul2 >> add1 >> mul2'
    assert supflow(42) == (42 * 2 + 1) * 2


def test_pop(flows):
    mul2, add1, supflow = flows

    assert len(supflow) == 2
    poped = supflow.pop()
    assert len(supflow) == 1
    assert poped is add1[0]

    assert supflow[0].name == 'mul2'
    assert supflow.name == 'mul2'
    assert supflow(42) == mul2(42) == 42 * 2


# ====================================================================================================
# Test Flow forward, chaining, selection


def test_forward(flows):
    mul2, add1, supflow = flows

    assert mul2(42) == 42 * 2
    assert add1(42) == 42 + 1
    assert supflow(42) == 42 * 2 + 1


def test_chaining(flows):
    _, _, supflow = flows

    assert supflow.name == 'mul2 >> add1'
    assert repr(supflow) == '<|mul2 >> add1>'
    assert len(supflow) == 2


def test_getitem(flows):
    mul2, add1, supflow = flows

    assert supflow[0] is mul2[0]
    assert supflow[1] is add1[0]

    assert supflow['mul2'] is mul2[0]
    assert supflow['add1'] is add1[0]


# ====================================================================================================
# Unit Test fixtures


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


# ====================================================================================================
