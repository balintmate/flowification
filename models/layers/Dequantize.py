import torch
import math


class Dequantize(torch.nn.Module):

    def forward(self, x, **kwargs):
        x += torch.rand(x.size()).to(x.device) / 256
        self.num_pixels = x.size(1)*x.size(2)*x.size(3)
        L = - self.num_pixels * math.log(256)
        return x, L

    def inverse(self, z, **kwargs):
        return z
