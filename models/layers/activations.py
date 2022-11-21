import torch
import torch.nn as nn
import nflows.transforms as T


class LeakyReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = 0.5

    def forward(self, x, **kwargs):
        # z = torch.where(x > 0, x, self.alpha * x)
        z = torch.nn.functional.leaky_relu(x, self.alpha, False)
        D = (x >= 0).float() + self.alpha * (x < 0).float()
        logL = D.log().reshape(x.size(0), -1).sum(1)
        return z, logL

    def inverse(self, z, **kwargs):
        x = torch.where(z > 0, z, z / self.alpha)
        return x


class ReconLeakyRelu(LeakyReLU):
    def forward(self, *args, **kwargs):
        return *super(ReconLeakyRelu, self).forward(*args, **kwargs), 0


class RqSpline(T.nonlinearities.PiecewiseRationalQuadraticCDF):

    def __init__(self, tail_bound=2., tails='linear', num_bins=10):
        super(RqSpline, self).__init__(
            1, tail_bound=tail_bound, tails=tails, num_bins=num_bins)

    def forward(self, inputs, context=None, **kwargs):
        sh = inputs.shape
        out, contr = super(RqSpline, self).forward(inputs.reshape(-1, 1))
        return out.view(sh), contr.view(sh[0], -1).sum(-1)

    def inverse(self, inputs, context=None, **kwargs):
        sh = inputs.shape
        out, contr = super(RqSpline, self).inverse(inputs.reshape(-1, 1))
        return out.view(sh)
