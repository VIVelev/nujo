from os.path import exists

import nujo.utils.data.nujo_dir as nujo_dir
from nujo.utils.data import Dataset
from nujo.utils.data.dataset_loader import DatasetLoader


def test_dataset_loader_download():
    DatasetLoader('iris', 'csv', True).download()

    assert exists(nujo_dir.HOME_DIR + 'iris.data')


def test_dataset_loader_install():
    dataset = Dataset('iris', 'csv', True, True)

    assert dataset.X.shape == (150, 4)
    assert dataset.y.shape == (150, 1)
