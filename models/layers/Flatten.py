import torch.nn as nn


class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, **kwargs):
        self.size = x.size()[1:]
        z = x.flatten(1)
        return z, 0

    def inverse(self, z, **kwargs):
        return z.view((z.size(0),) + self.size)


class ReconFlatten(Flatten):

    def forward(self, *args, **kwargs):
        return *super(ReconFlatten, self).forward(*args, **kwargs), 0
