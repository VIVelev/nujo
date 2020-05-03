import pytest

import nujo.optim as optim
from nujo import mean, rand, randn

# ====================================================================================================
# Test Stochastic Gradient Descent (SGD)


def test_sgd_basic(scalar_params, num_iters, quadratic_loss):
    def g():
        nonlocal scalar_params

        x = (yield scalar_params[0])
        scalar_params[0] <<= x
        yield

    optimizer = optim.SGD(g)

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

    prev_loss = 1e3
    for i in range(num_iters):
        loss = matrix_mse_loss(vec_params)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        assert loss < prev_loss
        prev_loss = loss


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

    prev_loss = 1e3
    for i in range(num_iters):
        loss = matrix_mse_loss(vec_params)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        assert loss < prev_loss
        prev_loss = loss


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

    prev_loss = 1e3
    for i in range(num_iters):
        loss = matrix_mse_loss(vec_params)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        assert loss < prev_loss
        prev_loss = loss


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

    prev_loss = 1e3
    for i in range(num_iters):
        loss = matrix_mse_loss(vec_params)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        assert loss < prev_loss
        prev_loss = loss


# ====================================================================================================
# PyTest Fixtures


@pytest.fixture
def scalar_params():
    return [rand(diff=True)]


@pytest.fixture
def vec_params():
    return [rand(3, 1, diff=True)]


@pytest.fixture
def num_iters():
    return 512


@pytest.fixture
def quadratic_loss():
    def compute(params):
        return 3 * (params[0]**2) + 5 * params[0] + 7

    return compute


@pytest.fixture
def matrix_mse_loss():
    X = rand(3, 3)
    y = X @ randn(3, 1) + rand()

    def compute(params):
        return mean((y - X @ params[0])**2)

    return compute


# ===================================================================================================
