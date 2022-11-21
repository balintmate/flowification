from torch import nn
import torch
from models.layers.Conv2d import Conv2d
from models.layers.Linear import Linear


# TODO this should just have the transform, the flow should be separate


class CompositeTransform(nn.Module):
    def __init__(self, list_of_layers):
        super(CompositeTransform, self).__init__()
        self.layers = nn.ModuleList(list_of_layers)

    def inverse(self, z, flag='mean'):
        assert flag in ['mean', 'sample']
        for Layer in self.layers[::-1]:
            z = Layer.inverse(z, flag=flag)
        return z

    def forward(self, x, flag='sample'):
        loglikelihood = 0
        for Layer in self.layers:
            x, L = Layer.forward(x, flag=flag)
            loglikelihood = loglikelihood + L
        return x, loglikelihood
