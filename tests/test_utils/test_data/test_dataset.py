from os.path import exists

import pytest

import nujo.utils.data.nujo_dir as nujo_dir
from nujo.utils.data import Dataset
from nujo.utils.data.dataset_loader import DatasetLoader


def test_dataset_loader_install(get_dataset):
    dataset = Dataset('iris')

    assert dataset.X.shape == get_dataset


def test_dataset_loader_download():
    DatasetLoader('iris').download()

    assert exists(nujo_dir.HOME_DIR + 'iris.data')


@pytest.fixture
def get_dataset():
    return 150, 5
