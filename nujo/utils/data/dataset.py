from nujo._typing import Union
from nujo.utils.data.dataset_iterator import DatasetIterator
from nujo.utils.data.dataset_loader import DatasetLoader, ndarray


class Dataset:
    ''' Dataset

    A class made for easy access and manipulation
    to online or local datasets

    Parameters:
    -----------
    name : will be downloaded from the UCI ML repo (for now)

    Returns:
    --------
    ndarray : stores the csv dataset,
    - floating point integers
    - last column -> labels

    '''
    def __init__(self, download: bool, name: str):
        self.name = name
        loader = DatasetLoader(self.name, type='csv')
        if download:
            loader.download()
        loader.install(self)

    def __iter__(self):
        return DatasetIterator(self)

    def __getitem__(self, position: Union[int, tuple]) -> ndarray:
        if isinstance(position, int):
            return self.X[position]
        row, col = position
        if (col != self._cols):
            # if not the last col, then return a float
            return float(self.X[row][col])
        # last col is a string
        return self.X[row][col]

    def __repr__(self):
        return self.X.__str__()
