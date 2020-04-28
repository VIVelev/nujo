from os import mkdir
from os.path import exists

from numpy import array, asarray, empty, vstack
from PIL import Image
from requests import get

from nujo.utils.data.nujo_dir import HOME_DIR


class DatasetLoader:
    '''

    Parameters:
    -----------
    name : will be downloaded from the UCI ML repo
    override : if this file exists, does it get downloaded again
    '''
    _UCI_REPO_URL = '''
    https://archive.ics.uci.edu/ml/machine-learning-databases/{}/{}
    '''

    def __init__(self, name: str, override=True):
        self.name = name  # with .data
        self._file = HOME_DIR + self.name
        if exists(HOME_DIR + name) and not override:
            return
        self._link = self._UCI_REPO_URL.format(
            self.name.split('.')[0], self.name).strip()
        self.download()

    def install(self, dataset, type: str) -> None:
        type = type.strip()
        # -----------------------------------------
        # reading csv
        if (type == 'csv'):
            with open(self._file, 'r+') as data:
                lines = data.readlines()
            dataset._cols = len(lines[0].split(','))
            dataset.X = empty((0, dataset._cols))
            # number of columns
            for line in lines[:-1]:  # last row is \n
                x = array(line.strip().split(','))
                dataset.X = vstack((dataset.X, x))
        # -----------------------------------------
        # reading image
        elif (type == 'image' or type == 'image' or type == 'png'
              or type == 'jpg'):
            # image has to be black and white
            dataset.X = empty((0, 0))
            with Image.open(self._file) as img:
                vect = asarray(img)
                assert vect.ndim < 3
                vect.reshape((vect.size, 1))
                dataset.X.append(vect)

    def download(self) -> None:
        r = get(self._link)
        if not exists(HOME_DIR):
            mkdir(HOME_DIR)
            print('Directory `~/.nujo` created')
        else:
            print('Directory `~/.nujo` already exists')
        print(f'File {self.name} has been created.')

        with open(self._file, 'wb') as f:
            f.write(r.content)
