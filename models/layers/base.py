from abc import ABC, abstractmethod

import torch.nn as nn


class Layer(ABC, nn.Module):

    @abstractmethod
    def forward(self, x, **kwargs):
        return 0

    @abstractmethod
    def inverse(self, z, **kwargs):
        return 0
