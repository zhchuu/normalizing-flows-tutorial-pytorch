'''
Reference:
https://github.com/kamenbliznashki/normalizing_flows/blob/master/maf.py
https://github.com/acids-ircam/pytorch_flows/blob/master/flows_04.ipynb
'''
import torch
from torch import nn
from nflows.base import Flow


class BatchNormFlow(Flow):

    def __init__(self, dim, momentum=0.95, eps=1e-5):
        super().__init__()
        # Running batch statistics
        self.r_mean = torch.zeros(dim)
        self.r_var = torch.ones(dim)
        # Momentum
        self.momentum = momentum
        self.eps = eps
        # Trainable scale and shift
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, z):
        if self.training:
            # Current batch stats
            self.b_mean = z.mean(0)
            self.b_var = (z - self.b_mean).pow(2).mean(0) + self.eps
            # Running mean and var
            self.r_mean = self.momentum * self.r_mean + ((1 - self.momentum) * self.b_mean)
            self.r_var = self.momentum * self.r_var + ((1 - self.momentum) * self.b_var)
            mean = self.b_mean
            var = self.b_var
        else:
            mean = self.r_mean
            var = self.r_var
        x_hat = (z - mean) / torch.sqrt(var + self.eps)
        x = self.gamma * x_hat + self.beta

        log_abs_det_jacobian = torch.log(self.gamma) - 0.5 * torch.log(var + self.eps)
        log_abs_det_jacobian = log_abs_det_jacobian.sum().unsqueeze(0).repeat(z.size(0), 1).squeeze()

        return x, log_abs_det_jacobian

    def inverse(self, x):
        if self.training:
            mean = self.b_mean
            var = self.b_var
        else:
            mean = self.r_mean
            var = self.r_var
        z_hat = (x - self.beta) / self.gamma
        z = z_hat * torch.sqrt(var + self.eps) + mean

        log_abs_det_jacobian = 0.5 * torch.log(var + self.eps) - torch.log(self.gamma)
        log_abs_det_jacobian = log_abs_det_jacobian.sum().unsqueeze(0).repeat(x.size(0), 1).squeeze()

        return z, log_abs_det_jacobian
