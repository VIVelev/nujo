from typing import List, Tuple, Union

from numpy import ndarray, pad

from nujo.autodiff.tensor import Tensor
from nujo.flow import Flow
from nujo.init import randn
from nujo.nn._transform import _flatten_image_sections, _get_image_section

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

        # May not be used
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation,
                                                                      dilation)

        self.apply_kernels = Linear(self.in_channels * self.kernel_size[0] *
                                    self.kernel_size[1],
                                    self.out_channels,
                                    bias=bias,
                                    name=f'{self.name}.kernels')

    def forward(self, x: Tensor) -> Tensor:
        # x is of shape (batch_size, channels, height, width)

        assert x.shape[1] == self.in_channels

        # pad the input images (before and after)
        x.value = pad(x.value, (
            (0, 0),
            (0, 0),
            (self.padding[0], self.padding[0]),
            (self.padding[1], self.padding[1]),
        ))

        # The following are stored, because they are
        # later used to compute the shape of the output
        batch_size = x.shape[0]
        height, width = x.shape[2], x.shape[3]

        sections: List[ndarray] = []
        for row_start in range(0, x.shape[2] - self.kernel_size[0] + 1,
                               self.stride[0]):
            for col_start in range(0, x.shape[3] - self.kernel_size[1] + 1,
                                   self.stride[1]):
                sections.append(
                    _get_image_section(x, row_start,
                                       row_start + self.kernel_size[0],
                                       col_start,
                                       col_start + self.kernel_size[1]))

        x.value = _flatten_image_sections(sections)  # flatten x
        flatten_output = self.apply_kernels(x)
        # the output need to be reshaped into volume
        return flatten_output.reshape(
            batch_size, self.out_channels,
            (height - self.kernel_size[0]) // self.stride[0] + 1,
            (width - self.kernel_size[1]) // self.stride[1] + 1)


# ====================================================================================================
