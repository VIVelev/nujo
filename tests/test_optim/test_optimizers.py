import pytest

import nujo.optim as optim
from nujo import mean, rand, randn

# ====================================================================================================
# Test Stochastic Gradient Descent (SGD)


def test_sgd_basic(params, num_iters, quadratic_loss):
    optimizer = optim.SGD(params)

    prev_loss = 1e3
    for _ in range(num_iters):
        loss = quadratic_loss(params)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        assert loss < prev_loss
        prev_loss = loss


def test_sgd_matrix(params, num_iters, matrix_mse_loss):
    optimizer = optim.SGD(params)

    moving_avrg_loss = 1e4
    for i in range(num_iters):
        loss = matrix_mse_loss(params)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if (i + 1) % 4 == 0:
            assert loss < moving_avrg_loss
        moving_avrg_loss = 0.9 * moving_avrg_loss + 0.1 * loss


# ====================================================================================================
# Test Momentum optimizer


def test_momentum_basic(params, num_iters, quadratic_loss):
    optimizer = optim.Momentum(params)

    prev_loss = 1e3
    for _ in range(num_iters):
        loss = quadratic_loss(params)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        assert loss < prev_loss
        prev_loss = loss


def test_momentum_matrix(params, num_iters, matrix_mse_loss):
    optimizer = optim.Momentum(params)

    moving_avrg_loss = 1e4
    for i in range(num_iters):
        loss = matrix_mse_loss(params)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if (i + 1) % 4 == 0:
            assert loss < moving_avrg_loss
        moving_avrg_loss = 0.9 * moving_avrg_loss + 0.1 * loss


# ====================================================================================================
# Test RMSprop optimizer


def test_rmsprop_basic(params, num_iters, quadratic_loss):
    optimizer = optim.RMSprop(params)

    prev_loss = 1e3
    for _ in range(num_iters):
        loss = quadratic_loss(params)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        assert loss < prev_loss
        prev_loss = loss


def test_rmsprop_matrix(params, num_iters, matrix_mse_loss):
    optimizer = optim.RMSprop(params)

    moving_avrg_loss = 1e4
    for i in range(num_iters):
        loss = matrix_mse_loss(params)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if (i + 1) % 4 == 0:
            assert loss < moving_avrg_loss
        moving_avrg_loss = 0.9 * moving_avrg_loss + 0.1 * loss


# ====================================================================================================
# Test Adam optimizer


def test_adam_basic(params, num_iters, quadratic_loss):
    optimizer = optim.Adam(params)

    prev_loss = 1e3
    for _ in range(num_iters):
        loss = quadratic_loss(params)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        assert loss < prev_loss
        prev_loss = loss


def test_adam_matrix(params, num_iters, matrix_mse_loss):
    optimizer = optim.Adam(params)

    moving_avrg_loss = 1e4
    for i in range(num_iters):
        loss = matrix_mse_loss(params)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if (i + 1) % 4 == 0:
            assert loss < moving_avrg_loss
        moving_avrg_loss = 0.9 * moving_avrg_loss + 0.1 * loss


# ====================================================================================================
# PyTest Fixtures


@pytest.fixture
def params():
    return [[rand()], [randn(3, 1)]]


@pytest.fixture
def num_iters():
    return 128


@pytest.fixture
def quadratic_loss():
    def compute(params):
        return 3 * (params[0][0]**2) + 5 * params[0][0] + 7

    return compute


@pytest.fixture
def matrix_mse_loss():
    X = rand(3, 3, diff=False)
    y = X @ randn(3, 1, diff=False) + rand(diff=False)

    def compute(params):
        return mean((y - X @ params[1][0])**2)

    return compute


# ===================================================================================================
