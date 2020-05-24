from typing import List

from numpy import concatenate, ndarray

from nujo.autodiff.tensor import Tensor

__all__ = [
    '_get_image_section',
]


def _get_image_section(image: Tensor, row_from: int, row_to: int,
                       col_from: int, col_to: int) -> ndarray:
    ''' Returns a subsection of an image (2d plane)

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
        (Batch size, 1, Channels, row_to - row_from, col_to - col_from)

    '''

    section = image.value[:, :, row_from:row_to, col_from:col_to]
    return section.reshape(-1, 1, image.shape[1], row_to - row_from,
                           col_to - col_from)


def _flatten_image_sections(sections: List[ndarray]) -> ndarray:
    extended = concatenate(sections, axis=1)
    return extended.reshape(extended.shape[0] * extended.shape[1], -1)
