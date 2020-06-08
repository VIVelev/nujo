import pytest
import torch
import torch.nn as torch_nn

import nujo as nj
import nujo.nn as nj_nn

# ====================================================================================================


def test_single_conv2d(image_input):
    nj_tensor, torch_tensor = image_input

    nj_conv = nj_nn.Conv2d(3, 6, 4, stride=2, padding=4, dilation=1)
    torch_conv = torch_nn.Conv2d(3, 6, 4, stride=2, padding=4, dilation=2)

    # Test Forward
    nj_output = nj_conv(nj_tensor)
    torch_output = torch_conv(torch_tensor)

    assert nj_output.shape == torch_output.shape

    # Test Backward
    nj_output.backward()


# ====================================================================================================
# Unit Test fixtures


@pytest.fixture
def image_input():
    shape = (32, 3, 28, 28)
    return nj.randn(*shape), torch.randn(*shape).float()


# ====================================================================================================
