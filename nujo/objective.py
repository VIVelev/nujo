from nujo.flow import Flow
from nujo.math import abs, mean, sum

__all__ = [
    'Loss',
    'L1Loss',
    'L2Loss',
]

# ====================================================================================================


class Loss(Flow):
    def __init__(self, dim=None, keepdim=False, reduction='mean'):
        super(Loss, self).__init__(name='<' + self.__class__.__name__ + '>')
        self.dim = dim
        self.keepdim = keepdim
        self.reduction = mean if reduction == 'mean' else sum


# ====================================================================================================


class L1Loss(Loss):
    def forward(self, input, target):
        return self.reduction(abs(input - target),
                              dim=self.dim,
                              keepdim=self.keepdim)


# ====================================================================================================


class L2Loss(Loss):
    def forward(self, input, target):
        return self.reduction((input - target)**2,
                              dim=self.dim,
                              keepdim=self.keepdim)


# ====================================================================================================
