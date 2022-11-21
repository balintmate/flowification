from models.layers.conv_kernel import Kernel, FancyKernel
from models.layers.conv_pad import Pad
from models.layers.conv_tile import Tile
import torch.nn as nn

# import wandb

import math
import sys


class Conv2d(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, stride):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        self.Pad = Pad(kernel_size, stride, augment="N")
        self.Tile = Tile(kernel_size, stride, augment="N")
        #self.kernel = Kernel(in_c, out_c, kernel_size, stride)
        self.kernel = FancyKernel(in_c, out_c, kernel_size, stride)

    def forward(self, x, flag="sample"):
        x_padded, L_pad = self.Pad(x, flag=flag)
        tiles, L_tile = self.Tile(x_padded, flag=flag)
        z, L_SVD = self.kernel(tiles, flag=flag)
        return z, L_pad + L_tile + L_SVD

    def inverse(self, z, flag):
        tiles = self.kernel.inverse(z, flag=flag)
        x_padded = self.Tile.inverse(tiles)
        x = self.Pad.inverse(x_padded)
        return x
