from numbers import Number
from typing import List, Optional, Tuple, Union

from numpy import add, arange, ndarray, pad, repeat, tile, zeros

from nujo._cache import cached_property
from nujo.autodiff.function import Function
from nujo.autodiff.tensor import Tensor

__all__ = [
    '_Reshape',
    '_Transpose',
    '_ConstPad',
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


class _ConstPad(Function):
    ''' Constant padding by a value

    Pads an array before and after for each dimension with a given constant
    value. (default: 0)

    Parameters:
    -----------
     - input : array to pad
     - padding : tuple of tuples of two ints specifying the padding before
     and after for each dimension
     - value : float, the constant value to pad with (default: 0)

    '''
    def __init__(self,
                 input: Union[Tensor, ndarray, List[Number], Number],
                 padding: Tuple[Tuple[int, int], ...],
                 value: float = 0):

        super(_ConstPad, self).__init__(input)

        # Assert a padding `(before, after)` was specified for each dimension
        # only. No more, no less.
        assert len(self.children[0].shape) == len(padding)

        self.padding = padding
        self.value = value

    def forward(self) -> ndarray:
        return pad(self.children[0].value,
                   self.padding,
                   constant_values=self.value)

    def backward(self, idx: int, accum_grad: Function.T) -> Function.T:
        idxs = tuple(
            slice(dim_pad[0], accum_grad.shape[i] - dim_pad[1])
            for i, dim_pad in enumerate(self.padding))

        return accum_grad[idxs]


# ====================================================================================================


class _Im2col(Function):
    ''' Image to column shape transformation

    The local regions in the input image are stretched out into columns.

    For example, if the input is [3x227x227] and it is to be convolved
    with 3x11x11 filters at stride (4, 4), then we would take [3x11x11]
    blocks of pixels in the input and stretch each block into a column
    vector of size 3*11*11 = 363. Iterating this process in the input
    at stride of (4, 4) gives (227-11)/4+1 = 55 locations along both
    height and width, leading to an output matrix X_col of Im2col of size
    [363 x 3025], where every column is a stretched out receptive field
    and there are 55*55 = 3025 of them in total.

    Reference: CS231n Stanford
    (https://cs231n.github.io/convolutional-networks/)

    Parameters:
    -----------
     - input : image shaped array, shape: (batch_size, channels, height, width)
     - kernel_size : tuple of 2 integers, image filter height and width
     - stride : tuple of 2 integers, stride of the convolution
     - dilation : tuple of 2 integers, spacing between kernel elements

    '''
    def __init__(
        self,
        input: Union[Tensor, ndarray, List[Number], Number],
        kernel_size: Tuple[int, int],
        stride: Tuple[int, int],
        dilation: Tuple[int, int],
    ):

        super(_Im2col, self).__init__(input)

        # Shape of `input` should be: (batch_size, channels, height, width)
        assert len(self.children[0].shape) == 4

        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation

    def forward(self) -> ndarray:
        ''' Method which turns the image shaped input to column shape
        '''

        images = self.children[0].value

        # Reshape content into column shape
        k, i, j = self._im2col_indices
        return images[:, k, i, j]\
            .transpose(1, 2, 0).reshape(self._n_features, -1)

    def backward(self, idx: int, accum_grad: Function.T) -> Function.T:
        ''' Method which turns the column shaped input to image shape
        '''

        # Create images placeholder
        images = zeros(self.children[0].shape)

        # Separate the image sections and the batch_size (shape[0])
        separated_grad = accum_grad\
            .reshape(self._n_features, -1, images.shape[0])\
            .transpose(2, 0, 1)  # Move the batch_size at the beginning

        # Fill in the placeholder
        k, i, j = self._im2col_indices
        add.at(images, (slice(None), k, i, j), separated_grad)

        return images

    @cached_property
    def _im2col_indices(self) -> Tuple[ndarray, ndarray, ndarray]:
        ''' Calculate the indices where the dot products are
        to be applied between the weights and the image.

        '''

        # Obtain needed information
        channels = self.children[0].shape[1]
        kernel_height, kernel_width = self.kernel_size
        stride_height, stride_width = self.stride
        dilation_height, dilation_width = self.dilation
        out_height, out_width = self._output_shape

        # Calculate sections' rows
        step = dilation_height + 1
        section_rows = repeat(arange(0, kernel_height * step, step),
                              kernel_width)
        section_rows = tile(section_rows, channels)

        # Slide rows by stride
        slide_rows = stride_width * repeat(arange(out_height), out_width)
        section_rows = section_rows.reshape(-1, 1) + slide_rows.reshape(1, -1)

        # Calculate sections' columns
        step = dilation_width + 1
        section_cols = tile(arange(0, kernel_width * step, step),
                            kernel_height * channels)

        # Slide cols by stride
        slide_cols = stride_height * tile(arange(out_width), out_height)
        section_cols = section_cols.reshape(-1, 1) + slide_cols.reshape(1, -1)

        # Calculate sections' channels
        section_channels = repeat(arange(channels),
                                  kernel_height * kernel_width).reshape(-1, 1)

        # Return indices
        return section_channels, section_rows, section_cols

    @cached_property
    def _output_shape(self):
        # Obtain needed information
        _, _, height, width = self.children[0].shape
        kernel_height, kernel_width = self.kernel_size
        stride_height, stride_width = self.stride
        dilation_height, dilation_width = self.dilation

        # Calculate output shape
        out_height = (height - dilation_height *
                      (kernel_height - 1) - kernel_height) // stride_height + 1
        out_width = (width - dilation_width *
                     (kernel_width - 1) - kernel_width) // stride_width + 1

        return out_height, out_width

    @cached_property
    def _n_features(self):
        ''' number of features in the column form
        '''

        return self.kernel_size[0] * self.kernel_size[1] *\
            self.children[0].shape[1]  # number of channels


# ====================================================================================================
