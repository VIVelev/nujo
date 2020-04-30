from os import mkdir
from os.path import exists

from numpy import array, asarray, empty, ndarray, vstack
from PIL import Image
from requests import get

from nujo.utils.data.nujo_dir import HOME_DIR


class DatasetLoader:
    '''

    Parameters:
    -----------
    - name : str,
    - type : str, indicates the type of file, can be csv, image or mnist
    - override : bool, if this file exists, does it get downloaded again
    '''
    _UCI_REPO_URL = '''
    https://archive.ics.uci.edu/ml/machine-learning-databases/{}/{}
    '''

    def __init__(self, name: str, type: str, override=True):
        self.name = name
        self._filepath = HOME_DIR + self.name
        if not override and exists(HOME_DIR + name):
            return
        self.type = self.type.strip().lower()

    def install(self,
                dataset,
                filepath: str = None,
                labels: ndarray = None) -> None:
        ''' Dataset Install\n
        (will override anything in the dataset)

        Parameters:
        -----------
        - dataset : Dataset, creates two ndarray containing vectors and labels
        - filepath : str, indicates source file, if none then `~/.nujo/`
        - labels : ndarray, labels for loading from image


        '''
        self._filepath = filepath if filepath is not None else self._filepath
        assert exists(self._filepath)

        # -----------------------------------------
        # reading csv
        if (type == 'csv'):
            with open(self._filepath, 'r+') as data:
                lines = data.readlines()
            dataset._cols = len(lines[0].split(','))
            dataset.X = empty((0, dataset._cols - 1))
            # number of columns
            for line in lines[:-1]:  # last row is \n
                x = array(line.strip().split(',')[:-2])
                dataset.X = vstack((dataset.X, x))
                dataset.y = vstack((dataset.y, line.strip().split(',')[-1]))
        # -----------------------------------------
        # reading image
        elif (type == 'image' or type == 'img' or type == 'png'
              or type == 'jpg'):
            assert labels is not None
            # image has to be black and white
            dataset.X = empty((0, 0))
            with Image.open(self._filepath) as img:
                vect = asarray(img)
                assert vect.ndim < 3
                vect.reshape((vect.size, 1))
                dataset.X.append(vect)
            dataset.y = labels

    def download(self) -> None:
        if not exists(HOME_DIR):
            mkdir(HOME_DIR)
            print('Directory `~/.nujo` created')

        if self.type == 'csv':
            self._link = self._UCI_REPO_URL.format(self.name,
                                                   f'{self.name}.data')
            with open(self._filepath, 'wb') as f:
                f.write(get(self._link).content)
            print(f'{self.name} has been saved in `~/.nujo`')
