from os import mkdir
from os.path import exists
from typing import Optional

from numpy import array, asarray, empty, ndarray, vstack
from PIL import Image
from requests import get

from nujo.utils.data.nujo_dir import HOME_DIR


class DatasetLoader:
    ''' Dataset Loader

    Parameters:
    -----------
     - name : str, downloaded from uci repo or filename to install from
     - type : str, indicates the type of file, can be csv, image or mnist
     - override : bool, if this file exists, does it get downloaded again

    '''

    _UCI_REPO_URL = (
        'https://archive.ics.uci.edu/ml/machine-learning-databases/{}/{}')

    def __init__(self, name: str, type: str, override=True):
        self.name = name.strip().lower()
        self.type = type.strip().lower()
        self._filepath = f'{HOME_DIR}{self.name}.data'

    def install(self,
                filepath: Optional[str] = None,
                labels: Optional[ndarray] = None) -> ndarray:
        ''' Dataset Install

        (will override anything in the Dataset class)

        Parameters:
        -----------
         - filepath : str, indicates source file, if none then `~/.nujo/`
         - labels : ndarray, labels for loading from image

        Returns:
        -----------
         - res : ndarray, X and y (data and labels)

        '''

        self._filepath = filepath if filepath is not None else self._filepath
        assert exists(self._filepath)

        # -----------------------------------------
        # reading csv
        if self.type == 'csv':
            with open(self._filepath, 'r+') as data:
                lines = data.readlines()

            cols = len(lines[0].split(','))
            X = empty((0, cols - 1))
            y = empty((0, 1))

            # number of columns
            for line in lines[:-1]:  # last row is \n
                X = vstack((X, array(line.strip().split(',')[:-1])))
                y = vstack((y, line.strip().split(',')[-1]))

        # -----------------------------------------
        # reading image
        elif (self.type == 'image' or self.type == 'img' or self.type == 'png'
              or self.type == 'jpg'):

            # image has to be black and white
            with Image.open(self._filepath) as img:
                vect = asarray(img)
                assert vect.ndim < 3
                X = vect.reshape((vect.size, 1))

            assert labels is not None
            y = labels

        else:
            raise ValueError("type should me csv or image")

        return X, y

    def download(self) -> None:
        if not exists(HOME_DIR):
            mkdir(HOME_DIR)
            print('Directory `~/.nujo` created')

        if self.type == 'csv':
            self._link = self._UCI_REPO_URL.format(self.name,
                                                   f'{self.name}.data')
            print(self._link)
            with open(self._filepath, 'wb') as f:
                f.write(get(self._link).content)
            print(f'{self.name} has been saved in `~/.nujo`')
