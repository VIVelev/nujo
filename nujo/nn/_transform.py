from typing import List

from numpy import concatenate, ndarray

from nujo.autodiff.tensor import Tensor

__all__ = [
    '_get_image_section',
    '_flatten_image_sections',
]


def _get_image_section(image: Tensor, row_from: int, row_to: int,
                       col_from: int, col_to: int) -> ndarray:
    ''' Returns a subsection of an image

    Parameters:
    -----------
    - image : Tensor of shape (Batch size, Channels, Hight, Width)
    - row_from : int
    - row_to : int
    - col_from : int
    - col_to : int

    Returns:
    --------
    - section : ndarray of shape:
        (Batch size, 1, Channels, `row_to - row_from`, `col_to - col_from`)

    '''

    section = image.value[:, :, row_from:row_to, col_from:col_to]
    return section.reshape(-1, 1, image.shape[1], row_to - row_from,
                           col_to - col_from)


def _flatten_image_sections(sections: List[ndarray]) -> ndarray:
    ''' Flatten sections of an image

    Flattens the sections of an image in a Tensor
    that can be passed to nn.Linear.

    Parameters:
    -----------
     - sections : list of ndarrays, each ndarray is of shape:
        (Batch size, 1, Channels, Hight, Width)

    Returns:
    --------
     - flatten : a Tensor of shape:
        (Batch size * len(sections), Channels * Hight * Width)

    '''

    grouped = concatenate(sections, axis=1)
    return grouped.reshape(grouped.shape[0] * grouped.shape[1], -1).T
