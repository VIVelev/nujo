import pytest

import nujo.optim as optim
from nujo import Tensor

# ====================================================================================================
# Test Stochastic Gradient Descent (SGD)


def test_sgd_basic(params, quadratic_loss):
    optimizer = optim.SGD(params)

    prev_loss = 1e6
    for _ in range(10):
        loss = quadratic_loss(params)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        assert loss < prev_loss
        prev_loss = loss


def test_sgd_matrix(params, matrix_mse_loss):
    optimizer = optim.SGD(params)

    prev_loss = 1e6
    for _ in range(10):
        loss = matrix_mse_loss(params)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        assert loss < prev_loss
        prev_loss = loss


# ====================================================================================================
# Test Momentum optimizer


def test_momentum_basic(params, quadratic_loss):
    optimizer = optim.Momentum(params)

    prev_loss = 1e6
    for _ in range(10):
        loss = quadratic_loss(params)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        assert loss < prev_loss
        prev_loss = loss


def test_momentum_matrix(params, matrix_mse_loss):
    optimizer = optim.Momentum(params)

    prev_loss = 1e6
    for _ in range(10):
        loss = matrix_mse_loss(params)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        assert loss < prev_loss
        prev_loss = loss


# ====================================================================================================
# Test RMSprop optimizer


def test_rmsprop_basic(params, quadratic_loss):
    optimizer = optim.RMSprop(params)

    prev_loss = 1e6
    for _ in range(10):
        loss = quadratic_loss(params)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        assert loss < prev_loss
        prev_loss = loss


def test_rmsprop_matrix(params, matrix_mse_loss):
    optimizer = optim.RMSprop(params)

    prev_loss = 1e6
    for _ in range(10):
        loss = matrix_mse_loss(params)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        assert loss < prev_loss
        prev_loss = loss


# ====================================================================================================
# Test Adam optimizer


def test_adam_basic(params, quadratic_loss):
    optimizer = optim.Adam(params)

    prev_loss = 1e6
    for _ in range(10):
        loss = quadratic_loss(params)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        assert loss < prev_loss
        prev_loss = loss


def test_adam_matrix(params, matrix_mse_loss):
    optimizer = optim.Adam(params)

    prev_loss = 1e6
    for _ in range(10):
        loss = matrix_mse_loss(params)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        assert loss < prev_loss
        prev_loss = loss


# ====================================================================================================


@pytest.fixture
def params():
    return [[Tensor(10)], [Tensor([[1], [2], [3]])]]


@pytest.fixture
def quadratic_loss():
    def compute(params):
        return 3 * (params[0][0]**2) + 5 * params[0][0] + 7

    return compute


# TODO: Finilize the `matrix_mse_loss`
# once nj.sum or nj.mean are implemented


@pytest.fixture
def matrix_mse_loss():
    X = Tensor(
        [  # [1, 2, 3],
            # [4, 5, 6],
            [7, 8, 9]
        ],
        diff=False)

    y = Tensor(
        [  # [10],
            # [11],
            [12]
        ],
        diff=False)

    def compute(params):
        return (y - X @ params[1][0])**2

    return compute


# ===================================================================================================
