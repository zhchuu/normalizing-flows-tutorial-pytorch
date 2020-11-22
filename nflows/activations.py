import torch
from torch import nn
from nflows.base import Flow

EPSILON = 1e-5


class LeakyReLU(Flow):
    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.randn([1]))
        nn.init.normal_(self.alpha, 0, 0.01)

    def forward(self, z):
        I = torch.ones_like(z)
        J = torch.where(z >= 0, I, torch.abs(self.alpha) * I)
        log_abs_det_jacobian = torch.sum(torch.log(torch.abs(J) + EPSILON), dim=1)
        return torch.where(z >= 0, z, torch.abs(self.alpha) * z), log_abs_det_jacobian

    def inverse(self, x):
        I = torch.ones_like(x)
        J = torch.where(x >= 0, I, torch.abs(1. / self.alpha) * I)
        log_abs_det_jacobian = torch.sum(torch.log(torch.abs(J) + EPSILON), dim=1)
        return torch.where(x >= 0, x, torch.abs(1. / self.alpha) * x), log_abs_det_jacobian
