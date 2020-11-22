import torch
from torch import nn
from nflows.base import Flow


class AffineTransform(Flow):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(dim, dim))
        self.shift = nn.Parameter(torch.randn(dim,))
        nn.init.orthogonal_(self.weight)

    def forward(self, z):
        log_abs_det_jacobian = torch.slogdet(self.weight)[-1].unsqueeze(0).repeat(z.size(0), 1).squeeze()
        return self.shift + z @ self.weight, log_abs_det_jacobian

    def inverse(self, x):
        log_abs_det_jacobian = torch.slogdet(torch.inverse(self.weight))[-1].unsqueeze(0).repeat(x.size(0), 1).squeeze()
        return (x - self.shift) @ torch.inverse(self.weight), log_abs_det_jacobian
