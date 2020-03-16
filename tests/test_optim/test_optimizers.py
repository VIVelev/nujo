import pytest

import nujo as nj
import nujo.optim as optim

# ====================================================================================================
# Test Stochastic Gradient Descent (SGD)


def test_sgd_basic(params, num_iters, quadratic_loss):
    optimizer = optim.SGD(params)

    prev_loss = 1e6
    for _ in range(num_iters):
        loss = quadratic_loss(params)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        assert loss < prev_loss
        prev_loss = loss


def test_sgd_matrix(params, num_iters, matrix_mse_loss):
    optimizer = optim.SGD(params)

    prev_loss = 1e6
    for _ in range(num_iters):
        loss = matrix_mse_loss(params)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        assert loss < prev_loss
        prev_loss = loss


# ====================================================================================================
# Test Momentum optimizer


def test_momentum_basic(params, num_iters, quadratic_loss):
    optimizer = optim.Momentum(params)

    prev_loss = 1e6
    for _ in range(num_iters):
        loss = quadratic_loss(params)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        assert loss < prev_loss
        prev_loss = loss


def test_momentum_matrix(params, num_iters, matrix_mse_loss):
    optimizer = optim.Momentum(params)

    prev_loss = 1e6
    for _ in range(num_iters):
        loss = matrix_mse_loss(params)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        assert loss < prev_loss
        prev_loss = loss


# ====================================================================================================
# Test RMSprop optimizer


def test_rmsprop_basic(params, num_iters, quadratic_loss):
    optimizer = optim.RMSprop(params)

    prev_loss = 1e6
    for _ in range(num_iters):
        loss = quadratic_loss(params)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        assert loss < prev_loss
        prev_loss = loss


def test_rmsprop_matrix(params, num_iters, matrix_mse_loss):
    optimizer = optim.RMSprop(params)

    prev_loss = 1e6
    for _ in range(num_iters):
        loss = matrix_mse_loss(params)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        assert loss < prev_loss
        prev_loss = loss


# ====================================================================================================
# Test Adam optimizer


def test_adam_basic(params, num_iters, quadratic_loss):
    optimizer = optim.Adam(params)

    prev_loss = 1e6
    for _ in range(num_iters):
        loss = quadratic_loss(params)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        assert loss < prev_loss
        prev_loss = loss


def test_adam_matrix(params, num_iters, matrix_mse_loss):
    optimizer = optim.Adam(params)

    prev_loss = 1e6
    for _ in range(num_iters):
        loss = matrix_mse_loss(params)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        assert loss < prev_loss
        prev_loss = loss


# ====================================================================================================


@pytest.fixture
def params():
    return [[nj.Tensor(10)],
            [nj.Tensor([[1], [2], [3]])]]


@pytest.fixture
def num_iters():
    return 100


@pytest.fixture
def quadratic_loss():
    def compute(params):
        return 3 * (params[0][0]**2) + 5 * params[0][0] + 7

    return compute


@pytest.fixture
def matrix_mse_loss():
    X = nj.Tensor([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]], diff=False)

    y = nj.Tensor([[10],
                   [11],
                   [12]], diff=False)

    def compute(params):
        return nj.mean((y - X @ params[1][0])**2)

    return compute


# ===================================================================================================
