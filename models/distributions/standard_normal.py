import math

# TODO Integrate these changes into the project
import torch

from models.distributions.base import Distribution


class LogNormal(Distribution):

    def __init__(self, dimension):
        super(LogNormal, self).__init__(dimension)

    def log_normal(self, x, mean=0):
        Z = -.5 * math.log(2 * math.pi)
        exp = -.5 * (x - mean) ** 2
        logN = (Z + exp).view(x.shape[0], -1).sum(1)
        return logN

    def _sample(self, num_samples, context=None):
        # return torch.randn(num_samples, self.dimension).to(self.device)
        if context is None:
            context = torch.zeros((1, self.dimension))
        return torch.normal(context, torch.ones((num_samples, 1)).to(context)).to(self.device)

    def _log_prob(self, x, context=None):
        return self.log_normal(x)
