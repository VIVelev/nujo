import pytest

import nujo.optim as optim
from nujo import mean, rand, randn

# ====================================================================================================
# Test Stochastic Gradient Descent (SGD)


@pytest.mark.slow
def test_sgd_basic(scalar_params, get_generator_for, num_iters,
                   quadratic_loss):

    optimizer = optim.SGD(get_generator_for(scalar_params))

    prev_loss = 1e3
    for _ in range(num_iters):
        loss = quadratic_loss(scalar_params)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        assert loss < prev_loss
        prev_loss = loss.value


@pytest.mark.slow
def test_sgd_matrix(vec_params, get_generator_for, num_iters, matrix_mse_loss):
    optimizer = optim.SGD(get_generator_for(vec_params))

    prev_loss = 1e3
    for i in range(num_iters):
        loss = matrix_mse_loss(vec_params)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        assert loss < prev_loss
        prev_loss = loss.value


# ====================================================================================================
# Test Momentum optimizer


@pytest.mark.slow
def test_momentum_basic(scalar_params, get_generator_for, num_iters,
                        quadratic_loss):

    optimizer = optim.Momentum(get_generator_for(scalar_params))

    prev_loss = 1e3
    for _ in range(num_iters):
        loss = quadratic_loss(scalar_params)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        assert loss < prev_loss
        prev_loss = loss.value


@pytest.mark.slow
def test_momentum_matrix(vec_params, get_generator_for, num_iters,
                         matrix_mse_loss):

    optimizer = optim.Momentum(get_generator_for(vec_params))

    prev_loss = 1e3
    for i in range(num_iters):
        loss = matrix_mse_loss(vec_params)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        assert loss < prev_loss
        prev_loss = loss.value


# ====================================================================================================
# Test RMSprop optimizer


@pytest.mark.slow
def test_rmsprop_basic(scalar_params, get_generator_for, num_iters,
                       quadratic_loss):

    optimizer = optim.RMSprop(get_generator_for(scalar_params))

    prev_loss = 1e3
    for _ in range(num_iters):
        loss = quadratic_loss(scalar_params)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        assert loss < prev_loss
        prev_loss = loss.value


@pytest.mark.slow
def test_rmsprop_matrix(vec_params, get_generator_for, num_iters,
                        matrix_mse_loss):

    optimizer = optim.RMSprop(get_generator_for(vec_params))

    prev_loss = 1e3
    for i in range(num_iters):
        loss = matrix_mse_loss(vec_params)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        assert loss < prev_loss
        prev_loss = loss.value


# ====================================================================================================
# Test Adam optimizer


@pytest.mark.slow
def test_adam_basic(scalar_params, get_generator_for, num_iters,
                    quadratic_loss):

    optimizer = optim.Adam(get_generator_for(scalar_params))

    prev_loss = 1e3
    for _ in range(num_iters):
        loss = quadratic_loss(scalar_params)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        assert loss < prev_loss
        prev_loss = loss.value


@pytest.mark.slow
def test_adam_matrix(vec_params, get_generator_for, num_iters,
                     matrix_mse_loss):

    optimizer = optim.Adam(get_generator_for(vec_params))

    prev_loss = 1e3
    for i in range(num_iters):
        loss = matrix_mse_loss(vec_params)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        assert loss < prev_loss
        prev_loss = loss.value


# ====================================================================================================
# PyTest Fixtures


@pytest.fixture
def scalar_params():
    return [rand(diff=True)]


@pytest.fixture
def vec_params():
    return [rand(3, 1, diff=True)]


@pytest.fixture
def get_generator_for():
    def gen(params):
        def g():
            yield params[0]

        return g

    return gen


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
