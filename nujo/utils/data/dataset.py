from typing import Tuple, Union

from numpy import ndarray

from nujo.utils.data.dataset_iterator import DatasetIterator
from nujo.utils.data.dataset_loader import DatasetLoader


class Dataset:
    ''' Dataset

    A class made for easy access and manipulation
    to online or local datasets

    Parameters:
    -----------
    - name : str, will be downloaded from the UCI ML repo (for now)
    - type : str, type of dataset - either csv or image
    - override : boolean, for loader, will override file if it exists
    - download : boolean, should it be downloaded from uci repo

    '''
    def __init__(self, name: str, type: str, override=True, download=False):
        self.name = name
        self.type = type
        loader = DatasetLoader(self.name, self.type, override)
        if download is True:
            loader.download()
        self.X, self.y = loader.install()

    def __iter__(self):
        return DatasetIterator(self)

    def __getitem__(self, position: Union[int, Tuple[int]]) -> ndarray:
        if isinstance(position, int):
            return self.X[position]
        row, col = position
        if (col != self._cols):
            return self.X[row][col]
        return self.y[row]

    def __repr__(self):
        return self.X.__str__()
