from os.path import exists

import pytest

import nujo.utils.data.nujo_dir as nujo_dir
from nujo.utils.data.dataset import Dataset
from nujo.utils.data.dataset_loader import DatasetLoader


@pytest.mark.slow
def test_dataset_loader_download():
    DatasetLoader('iris', 'csv', True).download()

    assert exists(nujo_dir.HOME_DIR + 'iris.data')


@pytest.mark.slow
def test_dataset_loader_install():
    dataset = Dataset('iris', 'csv', True, True)

    assert dataset.X.shape == (150, 4)
    assert dataset.y.shape == (150, 1)
