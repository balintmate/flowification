import torch
import torch.nn as nn
import math
from models.layers.Linear import Linear, Rotation


class Kernel(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        in_features = kernel_size[0] * kernel_size[1] * in_channels
        out_features = out_channels
        self.funnel = Linear(in_features, out_features)

    def forward(self, tiles, flag):
        z, logL = self.funnel(tiles, flag=flag)
        return z.permute(0, 3, 1, 2), logL.flatten(1).sum(1)

    def inverse(self, z, flag):
        z = z.permute(0, 2, 3, 1)
        tiles = self.funnel.inverse(z, flag=flag)
        return tiles


# inverse gathers local info to predict dropped coordinates
class FancyKernel(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.D1 = kernel_size[0] * kernel_size[1] * in_channels
        self.D2 = out_channels
        self.out_channels = out_channels
        if self.D2 >= self.D1:  # same as in Kernel
            self.Linear = Linear(self.D1, self.D2)
        else:
            self.U2 = Rotation(self.D2)
            self.b = torch.nn.Parameter(torch.zeros(1, self.D2))
            self.logW = torch.nn.Parameter(torch.zeros(1, self.D2))
            self.patchingInverse = PatchingInverse(in_channels,
                                                   out_channels,
                                                   kernel_size,
                                                   stride)
            # self.patchingInverse = PatchingInverseLocal(
            #     in_channels, out_channels, kernel_size, stride)

    def forward(self, tiles, flag):
        # tiles have sime (batch, x_loc, y_loc, in_c*kernel_size)
        # we apply a linear map over the last dimension and permute it to be the channels of the output image
        if self.D2 >= self.D1:  # same as in Kernel
            z, logL = self.Linear(tiles, flag)
            return z.permute(0, 3, 1, 2), logL.flatten(1).sum(1)
        else:
            x = self.patchingInverse.U1(tiles)
            x1, x2 = torch.split(x, [self.D1 - self.D2, self.D2], -1)
            z = torch.einsum("bxyc,ec->bxyc", (x2, self.logW.exp()))
            z = self.U2(z)
            z = z + self.b
            z = z.permute(0, 3, 1, 2)
            mu, logstd = self.patchingInverse(z)
            exp = -0.5 * (((mu - x1) / logstd.exp()) ** 2)
            Z = -0.5 * math.log(2 * math.pi) - logstd
            logpx1 = (exp + Z).flatten(1).sum(1)
            return z, self.logW.sum(1) * z.size(-1) * z.size(-2) + logpx1

    def inverse(self, z, flag):
        if self.D2 >= self.D1:  # same as in Kernel
            z = z.permute(0, 2, 3, 1)
            tiles = self.Linear.inverse(z, flag=flag)
            return tiles
        else:
            mu, logstd = self.patchingInverse(z)
            if flag == "mean":
                x1 = mu  # +logstd.exp()*randn
            elif flag == "sample":
                x1 = mu + logstd.exp() * torch.randn(mu.size()).to(mu)
            z = z.permute(0, 2, 3, 1)
            z = z - self.b
            z = self.U2.inverse(z)
            x2 = torch.einsum("bxyc,ec->bxyc", (z, (-self.logW).exp()))
            x = torch.cat([x1, x2], -1)
            tiles = self.patchingInverse.U1.inverse(x)
            return tiles


class PatchingInverse(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        # (kernel_size - 1) - padding
        self.D1 = kernel_size[0] * kernel_size[1] * in_channels
        self.D2 = out_channels
        self.U1 = Rotation(self.D1)
        self.kernel_size = kernel_size
        self.stride = stride
        self.in_channels = in_channels

        N = 2 * in_channels + out_channels
        self.MLP = nn.Sequential(  # apply this per pixel
            nn.Linear(out_channels, N), nn.ReLU(),
            nn.Linear(N, N), nn.ReLU(),
            nn.Linear(N, N), nn.ReLU(),
            nn.Linear(N, N), nn.ReLU(),
            nn.Linear(N, N), nn.ReLU(),
            nn.Linear(N, N),
        )
        self.MLP2 = nn.Sequential(  # apply this per pixel
            nn.Linear(N, N), nn.ReLU(),
            nn.Linear(N, N), nn.ReLU(),
            nn.Linear(N, N), nn.ReLU(),
            nn.Linear(N, N), nn.ReLU(),
            nn.Linear(N, N), nn.ReLU(),
            nn.Linear(N, 2 * in_channels),
        )
        # this will make a tensor that matches the size of the input
        self.Conv = nn.Sequential(
            nn.ConvTranspose2d(N, N, kernel_size=kernel_size, stride=stride),
            nn.ReLU(), nn.Conv2d(N, N, kernel_size=3, stride=1, padding=1),
            nn.ReLU(), nn.Conv2d(N, N, kernel_size=3, stride=1, padding=1)
        )

    def Net(self, z):
        z = z.permute(0, 2, 3, 1)
        z = self.MLP(z)
        z = z.permute(0, 3, 1, 2)
        z = self.Conv(z)
        z = z.permute(0, 2, 3, 1)
        z = self.MLP2(z)
        x = z.permute(0, 3, 1, 2)
        return x

    def unfold(self, x):
        tiles = nn.Unfold(kernel_size=self.kernel_size,
                          stride=self.stride)(x).transpose(-2, -1)
        return tiles

    def forward(self, z):
        mu_x, logstd_x = self.Net(z).split(self.in_channels, 1)  # image space
        mu_tiles = self.unfold(mu_x)
        logstd_tiles = self.unfold(logstd_x)

        # count patches
        num_X = 1 + (mu_x.size(2) - self.kernel_size[0]) // self.stride[0]
        num_Y = 1 + (mu_x.size(3) - self.kernel_size[1]) // self.stride[1]
        mu_tiles = mu_tiles.view(mu_tiles.size(0), num_X, num_Y, -1)
        logstd_tiles = logstd_tiles.view(logstd_tiles.size(0),
                                         num_X, num_Y, -1)

        mu_rotated_tiles = self.U1(mu_tiles)
        logstd_rotated_tiles = self.U1(logstd_tiles)

        return (
            mu_rotated_tiles[:, :, :, : self.D1 - self.D2],
            logstd_rotated_tiles[:, :, :, : self.D1 - self.D2],
        )
