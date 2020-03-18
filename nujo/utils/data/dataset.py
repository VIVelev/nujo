from numpy import array

from nujo.utils.data.nujo_dir import HOME_DIR
from nujo.utils.data.dataset_iterator import DatasetIterator
from nujo.utils.data.dataset_loader import DatasetLoader


class Dataset:
    ''' Dataset

    A class made for easy access and manipulation
    to online or local datasets

    Parameters:
    -----------
    name : will be downloaded from the UCI ML repo (for now)

    Returns:
    --------
    array : stores the csv dataset,
    - floating point integers
    - last column -> labels

    '''
    def __init__(self, name: str):
        self.name = HOME_DIR + name
        if '.data' not in name:
            self.name += '.data'
        DatasetLoader(self)

    def __iter__(self):
        return DatasetIterator(self)

    def __getitem__(self, position) -> array:
        if isinstance(position, int):
            return self.X[position]
        row, col = position
        return self.X[row][col]


if __name__ == '__main__':
    data = Dataset('iris')
    print(data[2])
