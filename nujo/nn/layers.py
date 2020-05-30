from typing import Tuple, Union

from numpy import arange, repeat, tile

from nujo.autodiff.tensor import Tensor
from nujo.flow import Flow
from nujo.init import randn

__all__ = [
    'Linear',
    'Conv2d',
]

# ====================================================================================================


class Linear(Flow):
    ''' Linear Layer

        f(x) = Wx + b

    Parameters:
    -----------
     - in_features : int, dim of input variables
     - out_features : int, wanted dim of output variables
     - bias : bool, whether to train a bias term or no
     - name : string, identifier for the current layer

    '''
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias=True,
                 name='Linear'):

        super(Linear,
              self).__init__(name=f'{name}({in_features}, {out_features})')

        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

        self.W = randn(self.out_features,
                       self.in_features,
                       name=self.name + '.W')

        if self.bias:
            self.b = randn(self.out_features, 1, name=self.name + '.bias')

    def forward(self, x: Tensor) -> Tensor:
        out = self.W @ x
        return out + self.b if self.bias else out


# ====================================================================================================


class Conv2d(Flow):
    ''' A 2-dimensional convolutional layer

    Applies a 2D convolution over an input signal composed of
    several input planes.
    More info: https://cs231n.github.io/convolutional-networks/

    Parameters:
    -----------
     - in_channels : int, number of channels in the input image
     - out_channels : int, number of channels produced by the convolution
        (in other word, the number of kernels)
     - kernel_size : int or tuple, size of the convolving kernel
     - stride : int or tuple, optional, stride of the convolution. Default: 1
     - padding : int or tuple, optional, zero-padding added to both sides of
        the input. Default: 0
     - dilation : int or tuple, optional - spacing between kernel elements.
        Default: 0
     - bias : bool, optional, if True, adds a learnable bias to the output.
        Default: True

    '''
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[int, Tuple[int, int]] = 0,
                 dilation: Union[int, Tuple[int, int]] = 0,
                 bias=True,
                 name='Conv2d'):

        super(Conv2d,
              self).__init__(name=f'{name}({in_channels}, {out_channels})')

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.kernel_size = kernel_size if isinstance(
            kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding,
                                                                   padding)

        # Not used for now
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation,
                                                                      dilation)

        self.kernels = randn(self.out_channels,
                             self.in_channels,
                             *self.kernel_size,
                             name=self.name + '.kernels')

    def forward(self, x: Tensor) -> Tensor:
        batch_size, channels, height, width = x.shape
        assert channels == self.in_channels

        # Turn image shape into column shape
        # (enables dot product between input and weights)
        pass


def get_im2col_indices(images_shape, kernel_size, stride):
    ''' Reference: CS231n Stanford
    (https://cs231n.github.io/convolutional-networks/)

    '''

    # Obtain needed  information
    _, channels, height, width = images_shape
    kernel_height, kernel_width = kernel_size

    # Calculate output shape
    out_height = (height - kernel_height) // stride + 1
    out_width = (width - kernel_width) // stride + 1

    # Calculate sections' rows
    section_rows = repeat(arange(kernel_height), kernel_width)
    section_rows = tile(section_rows, channels)
    slide_rows = stride * repeat(arange(out_height), out_width)
    section_rows = section_rows.reshape(-1, 1) + slide_rows.reshape(1, -1)

    # Calculate sections' columns
    section_cols = tile(arange(kernel_width), kernel_height * channels)
    slide_cols = stride * tile(arange(out_width), out_height)
    section_cols = section_cols.reshape(-1, 1) + slide_cols.reshape(1, -1)

    # Calculate sections' channels
    section_channels = repeat(arange(channels),
                              kernel_height * kernel_width).reshape(-1, 1)

    # Return indices
    return section_channels, section_rows, section_cols


def im2col(images, kernel_size, stride):
    ''' Method which turns the image shaped input to column shape.
    Used during the forward pass.

    Reference: CS231n Stanford
    (https://cs231n.github.io/convolutional-networks/)

    '''

    # Calculate the indices where the dot products are
    # to be applied between weights and the image
    k, i, j = get_im2col_indices(images.shape, kernel_size, stride)

    # Reshape content into column shape
    return images[:, k, i, j].transpose(1, 2, 0)\
        .reshape(kernel_size[0] * kernel_size[1] * images.shape[1], -1)


# ====================================================================================================
