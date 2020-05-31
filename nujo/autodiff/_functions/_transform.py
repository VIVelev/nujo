from numbers import Number
from typing import List, Optional, Tuple, Union

from numpy import add, arange, expand_dims, ndarray, pad, repeat, tile, zeros

from nujo.autodiff.function import Function
from nujo.autodiff.tensor import Tensor

__all__ = [
    '_Reshape',
    '_Transpose',
    '_Pad',
    '_Im2col',
]

# ====================================================================================================


class _Reshape(Function):
    def __init__(self, input: Union[Tensor, ndarray, List[Number], Number],
                 shape: Tuple[int, ...]):

        super(_Reshape, self).__init__(input)

        self.shape = shape
        self._input_shape = self.children[0].shape

    def forward(self) -> ndarray:
        return self.children[0].value.reshape(*self.shape)

    def backward(self, idx: int, accum_grad: Function.T) -> Function.T:
        return accum_grad.reshape(*self._input_shape)


# ====================================================================================================


class _Transpose(Function):
    def __init__(self,
                 input: Union[Tensor, ndarray, List[Number], Number],
                 dims: Optional[Tuple[int, ...]] = None):

        super(_Transpose, self).__init__(input)

        self.dims = dims if dims is not None else reversed(
            range(len(self.children[0].shape)))
        self._detranspose_dims = sorted(range(len(self.dims)),
                                        key=lambda idx: self.dims[idx])

    def forward(self) -> ndarray:
        return self.children[0].value.transpose(*self.dims)

    def backward(self, idx: int, accum_grad: Function.T) -> Function.T:
        return accum_grad.transpose(*self._detranspose_dims)


# ====================================================================================================


class _Pad(Function):
    def __init__(self, input: Union[Tensor, ndarray, List[Number], Number],
                 padding: Tuple[int, int]):

        super(_Pad, self).__init__(input)

        # Shape of `input` should be: (batch_size, channels, height, width)
        assert len(self.children[0].shape) == 4

        self.padding = padding

    def forward(self) -> ndarray:
        return pad(self.children[0].value, (
            (0, 0),
            (0, 0),
            (self.padding[0], self.padding[0]),
            (self.padding[1], self.padding[1]),
        ))

    def backward(self, idx: int, accum_grad: Function.T) -> Function.T:
        return accum_grad[:, :, self.padding[0]:-self.padding[0],
                          self.padding[1]:-self.padding[1]]


# ====================================================================================================


class _Im2col(Function):
    ''' Image to columns transformation

    Reference: CS231n Stanford
    (https://cs231n.github.io/convolutional-networks/)

    Parameters:
    -----------
     - input : image shaped array, shape: (batch_size, channels, height, width)
     - kernel_size : tuple of 2 integers, image filter height and width
     - stride : tuple of 2 integers

    '''
    def __init__(self, input: Union[Tensor, ndarray, List[Number], Number],
                 kernel_size: Tuple[int, int], stride: Tuple[int, int]):

        super(_Im2col, self).__init__(input)

        # Shape of `input` should be: (batch_size, channels, height, width)
        assert len(self.children[0].shape) == 4

        self.kernel_size = kernel_size

        # Calculate the indices where the dot products are
        # to be applied between weights and the image
        self._im2col_indices: Tuple[ndarray, ndarray, ndarray] =\
            self._get_im2col_indices(self.kernel_size, stride)

    def forward(self) -> ndarray:
        ''' Method which turns the image shaped input to column shape
        '''

        images = self.children[0].value

        # Reshape content into column shape
        n_features = self.kernel_size[0] * self.kernel_size[1] *\
            images.shape[1]  # number of channels

        k, i, j = self._im2col_indices
        return images[:, k, i, j].transpose(1, 2, 0).reshape(n_features, -1)

    def backward(self, idx: int, accum_grad: Function.T) -> Function.T:
        ''' Method which turns the column shaped input to image shape
        '''

        images = zeros(self.children[0].shape)

        k, i, j = self._im2col_indices
        add.at(images, (slice(None), k, i, j),
               expand_dims(accum_grad, 1).transpose(2, 0, 1))

        return images

    def _get_im2col_indices(
            self, kernel_size: Tuple[int, int],
            stride: Tuple[int, int]) -> Tuple[ndarray, ndarray, ndarray]:

        # Obtain needed  information
        _, channels, height, width = self.children[0].shape
        kernel_height, kernel_width = kernel_size
        stride_height, stride_width = stride

        # Calculate output shape
        out_height = (height - kernel_height) // stride_height + 1
        out_width = (width - kernel_width) // stride_width + 1

        # Calculate sections' rows
        section_rows = repeat(arange(kernel_height), kernel_width)
        section_rows = tile(section_rows, channels)
        slide_rows = stride_width * repeat(arange(out_height), out_width)
        section_rows = section_rows.reshape(-1, 1) + slide_rows.reshape(1, -1)

        # Calculate sections' columns
        section_cols = tile(arange(kernel_width), kernel_height * channels)
        slide_cols = stride_height * tile(arange(out_width), out_height)
        section_cols = section_cols.reshape(-1, 1) + slide_cols.reshape(1, -1)

        # Calculate sections' channels
        section_channels = repeat(arange(channels),
                                  kernel_height * kernel_width).reshape(-1, 1)

        # Return indices
        return section_channels, section_rows, section_cols


# ====================================================================================================
