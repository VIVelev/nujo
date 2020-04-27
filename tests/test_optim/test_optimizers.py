import pytest

import nujo.optim as optim
from nujo import mean, rand, randn

# ====================================================================================================
# Test Stochastic Gradient Descent (SGD)


def test_sgd_basic(scalar_params, num_iters, quadratic_loss):
    optimizer = optim.SGD(scalar_params)

    prev_loss = 1e3
    for _ in range(num_iters):
        loss = quadratic_loss(scalar_params)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        assert loss < prev_loss
        prev_loss = loss


def test_sgd_matrix(vec_params, num_iters, matrix_mse_loss):
    optimizer = optim.SGD(vec_params)

    moving_avrg_loss = 1e4
    for i in range(num_iters):
        loss = matrix_mse_loss(vec_params)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if (i + 1) % 4 == 0:
            assert loss < moving_avrg_loss
        moving_avrg_loss = 0.9 * moving_avrg_loss + 0.1 * loss


# ====================================================================================================
# Test Momentum optimizer


def test_momentum_basic(scalar_params, num_iters, quadratic_loss):
    optimizer = optim.Momentum(scalar_params)

    prev_loss = 1e3
    for _ in range(num_iters):
        loss = quadratic_loss(scalar_params)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        assert loss < prev_loss
        prev_loss = loss


def test_momentum_matrix(vec_params, num_iters, matrix_mse_loss):
    optimizer = optim.Momentum(vec_params)

    moving_avrg_loss = 1e4
    for i in range(num_iters):
        loss = matrix_mse_loss(vec_params)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if (i + 1) % 4 == 0:
            assert loss < moving_avrg_loss
        moving_avrg_loss = 0.9 * moving_avrg_loss + 0.1 * loss


# ====================================================================================================
# Test RMSprop optimizer


def test_rmsprop_basic(scalar_params, num_iters, quadratic_loss):
    optimizer = optim.RMSprop(scalar_params)

    prev_loss = 1e3
    for _ in range(num_iters):
        loss = quadratic_loss(scalar_params)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        assert loss < prev_loss
        prev_loss = loss


def test_rmsprop_matrix(vec_params, num_iters, matrix_mse_loss):
    optimizer = optim.RMSprop(vec_params)

    moving_avrg_loss = 1e4
    for i in range(num_iters):
        loss = matrix_mse_loss(vec_params)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if (i + 1) % 4 == 0:
            assert loss < moving_avrg_loss
        moving_avrg_loss = 0.9 * moving_avrg_loss + 0.1 * loss


# ====================================================================================================
# Test Adam optimizer


def test_adam_basic(scalar_params, num_iters, quadratic_loss):
    optimizer = optim.Adam(scalar_params)

    prev_loss = 1e3
    for _ in range(num_iters):
        loss = quadratic_loss(scalar_params)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        assert loss < prev_loss
        prev_loss = loss


def test_adam_matrix(vec_params, num_iters, matrix_mse_loss):
    optimizer = optim.Adam(vec_params)

    moving_avrg_loss = 1e4
    for i in range(num_iters):
        loss = matrix_mse_loss(vec_params)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if (i + 1) % 4 == 0:
            assert loss < moving_avrg_loss
        moving_avrg_loss = 0.9 * moving_avrg_loss + 0.1 * loss


# ====================================================================================================
# PyTest Fixtures


@pytest.fixture
def scalar_params():
    return [[rand()]]


@pytest.fixture
def vec_params():
    return [[randn(3, 1)]]


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
        return mean((y - X @ params[0][0])**2, inplace=True)

    return compute


# ===================================================================================================
