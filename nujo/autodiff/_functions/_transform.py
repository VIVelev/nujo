from numbers import Number
from typing import List, Optional, Tuple, Union

from numpy import add, arange, expand_dims, ndarray, repeat, tile, zeros

from nujo.autodiff.function import Function
from nujo.autodiff.tensor import Tensor

__all__ = [
    '_Reshape',
    '_Transpose',
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
        return self.children[0].value.reshape(self.shape)

    def backward(self, idx: int, accum_grad: Function.T) -> Function.T:
        return accum_grad.reshape(self._input_shape)


# ====================================================================================================


class _Transpose(Function):
    def __init__(self,
                 input: Union[Tensor, ndarray, List[Number], Number],
                 dims: Optional[Tuple[int, ...]] = None):

        super(_Transpose, self).__init__(input)

        self.dims = dims if dims is not None else reversed(
            range(len(self.dims)))
        self._detranspose_dims = sorted(range(len(self.dims)),
                                        key=lambda idx: self.dims[idx])

    def forward(self) -> ndarray:
        return self.children[0].value.transpose(*self.dims)

    def backward(self, idx: int, accum_grad: Function.T) -> Function.T:
        return accum_grad.transpose(*self._detranspose_dims)


# ====================================================================================================


class _Im2col(Function):
    def __init__(self, input: Union[Tensor, ndarray, List[Number], Number],
                 kernel_size: Tuple[int, int], stride: Tuple[int, int]):

        super(_Im2col, self).__init__(input)

        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self) -> ndarray:
        ''' Method which turns the image shaped input to column shape

        Reference: CS231n Stanford
        (https://cs231n.github.io/convolutional-networks/)

        '''

        images = self.children[0].value

        # Calculate the indices where the dot products are
        # to be applied between weights and the image
        k, i, j = _Im2col.get_im2col_indices(images.shape, self.kernel_size,
                                             self.stride)

        # Reshape content into column shape
        n_features = self.kernel_size[0] * self.kernel_size[1] *\
            images.shape[1]  # number of channels

        return images[:, k, i, j].transpose(1, 2, 0).reshape(n_features, -1)

    def backward(self, idx: int, accum_grad: Function.T) -> Function.T:
        ''' Method which turns the column shaped input to image shape

        Reference: CS231n Stanford
        (https://cs231n.github.io/convolutional-networks/)

        '''

        images_shape = self.children[0].shape

        # Calculate the indices where the dot products are
        # to be applied between weights and the image
        k, i, j = _Im2col.get_im2col_indices(images_shape, self.kernel_size,
                                             self.stride)

        images = zeros(images_shape)
        add.at(images, (slice(None), k, i, j),
               expand_dims(accum_grad, 1).transpose(2, 0, 1))

        return images

    @classmethod
    def get_im2col_indices(cls, images_shape, kernel_size, stride):
        ''' Reference: CS231n Stanford
        (https://cs231n.github.io/convolutional-networks/)

        '''

        # Obtain needed  information
        _, channels, height, width = images_shape
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
