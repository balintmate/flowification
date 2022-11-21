
import torch.nn as nn
import torch
import math


class Pad(nn.Module):
    def __init__(self, kernel_size, stride, augment):
        super().__init__()
        assert augment in ["U", "N"]
        self.augment = augment
        self.kernel_size = kernel_size
        self.stride = stride
        self.logA = nn.Parameter(torch.randn(1,))

    def log_normal(self, x):
        Z = -0.5 * math.log(2 * math.pi) - self.logA
        exp = -0.5 * (x / self.logA.exp()) ** 2
        logN = Z + exp
        return logN.view(logN.size(0), -1).sum(1)

    def forward(self, x, flag):
        s = self.stride
        im = x.size()[-2:]
        k = self.kernel_size
        pad_right = (s[0] - ((im[0] - k[0]) % s[0])) % s[0]
        pad_bottom = (s[1] - ((im[1] - k[1]) % s[1])) % s[1]

        self.size_of_last_input = x.size()[2:]
        if flag == "sample":
            A = self.logA.exp()
        elif flag == "mean":
            A = 0

        right_pad_size = (x.size()[:2]) + (pad_right, x.size(3))
        bottom_pad_size = (x.size()[:2]) + (x.size(2)+pad_right, pad_bottom)

        if self.augment == 'U':
            R_padding = A * (torch.rand(right_pad_size).to(x) - 0.5)
            B_padding = A * (torch.rand(bottom_pad_size).to(x) - 0.5)
            orig_pixels = x.size(-2) * x.size(-1)
            new_pixels = (x.size(-2)+pad_right) * (x.size(-1)+pad_bottom)
            L_augment = -self.logA * x.size(1) * (new_pixels-orig_pixels)
        elif self.augment == 'N':
            R_padding = A * (torch.randn(right_pad_size).to(x))
            B_padding = A * (torch.randn(bottom_pad_size).to(x))
            L_augment = self.log_normal(R_padding) + self.log_normal(B_padding)

        x = torch.cat((x, R_padding), -2)
        x = torch.cat((x, B_padding), -1)

        return x, -L_augment

    def inverse(self, z):
        return z[:, :, : self.size_of_last_input[0], : self.size_of_last_input[1]]
