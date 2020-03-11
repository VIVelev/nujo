import pytest

import nujo.optim as optim
from nujo import Tensor

# ====================================================================================================
# Test Stochastic Gradient Descent (SGD)


def test_sgd(params, compute_quadratic_loss):
    optimizer = optim.SGD(params)

    prev_loss = 1e6
    for _ in range(10):
        loss = compute_quadratic_loss(params)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        assert loss < prev_loss
        prev_loss = loss


# ====================================================================================================
# Test Momentum optimizer


def test_momentum(params, compute_quadratic_loss):
    optimizer = optim.Momentum(params)

    prev_loss = 1e6
    for _ in range(10):
        loss = compute_quadratic_loss(params)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        assert loss < prev_loss
        prev_loss = loss


# ====================================================================================================
# Test RMSprop optimizer


def test_rmsprop(params, compute_quadratic_loss):
    optimizer = optim.RMSprop(params)

    prev_loss = 1e6
    for _ in range(10):
        loss = compute_quadratic_loss(params)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        assert loss < prev_loss
        prev_loss = loss


# ====================================================================================================
# Test Adam optimizer


def test_adam(params, compute_quadratic_loss):
    optimizer = optim.Adam(params)

    prev_loss = 1e6
    for _ in range(10):
        loss = compute_quadratic_loss(params)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        assert loss < prev_loss
        prev_loss = loss


# ====================================================================================================


@pytest.fixture
def params():
    return [[Tensor(10)]]


@pytest.fixture
def compute_quadratic_loss():
    def compute(params):
        return 3 * (params[0][0]**2) + 5 * params[0][0] + 7

    return compute


# ===================================================================================================
