from torch import nn


class Flow(nn.Module):
    '''
    Basic flow structure
    '''
    def __init__(self):
        super(Flow, self).__init__()

    def forward(self, z):
        '''
        Args:
        z (tensor): samples of the source distribution with size (batch_size, )

        Returns:
        (tensors): transformed samples of the target distribution with size (batch_size, )
        '''
        pass

    def inverse(self, x):
        '''
        Args:
        x (tensor): samples of the target distribution with size (batch_size, )

        Returns:
        (tensors): transformed samples of the source distribution with size (batch_size, )
        '''
        pass


class FlowSequential(nn.Sequential):
    def forward(self, z):
        xs = [z]
        sum_log_abs_det_jacobians = 0
        for module in self:
            z, log_abs_det_jacobian = module(z)
            xs.append(z)
            sum_log_abs_det_jacobians = sum_log_abs_det_jacobians + log_abs_det_jacobian
        return xs, sum_log_abs_det_jacobians

    def inverse(self, x):
        zs = [x]
        sum_log_abs_det_jacobians = 0
        for module in reversed(self):
            x, log_abs_det_jacobian = module.inverse(x)
            zs.append(x)
            sum_log_abs_det_jacobians = sum_log_abs_det_jacobians + log_abs_det_jacobian

        return zs, sum_log_abs_det_jacobians
