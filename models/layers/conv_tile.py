import torch
import math
import sys


class Tile(torch.nn.Module):
    def __init__(self, kernel_size, stride, augment):
        super().__init__()
        assert augment in ["U", "N"]
        self.augment = augment
        self.kernel_size = kernel_size
        self.stride = stride
        self.logA = torch.nn.Parameter(torch.tensor([[.1]]).log())

    def unfold(self, x):
        tiles = torch.nn.Unfold(kernel_size=self.kernel_size,
                                stride=self.stride)(x).transpose(-2, -1)
        return tiles

    def log_normal(self, x):
        Z = -0.5 * math.log(2 * math.pi) - self.logA
        exp = -0.5 * (x / self.logA.exp()) ** 2
        logN = Z + exp
        return logN

    def init_params(self, x):
        if not hasattr(self, "N"):
            self.last_input_size = x.size()[-2:]
            ones = torch.ones((1, 1) + self.last_input_size).to(x)
            self.N = torch.nn.Fold(output_size=self.last_input_size,
                                   kernel_size=self.kernel_size,
                                   stride=self.stride,
                                   )(self.unfold(ones).transpose(-2, -1))
            self.N_allC = self.N.repeat(1, x.size(1), 1, 1)

            if self.augment == 'U':
                pad_dim = self.N-1
                logV = (pad_dim)/2*math.log(math.pi) - \
                    torch.lgamma((pad_dim)/2+1)
                self.unit_loglikelihoods = -logV

            elif self.augment == 'N':

                self.chi2 = torch.distributions.chi2.Chi2(
                    torch.max((self.N_allC - 1),
                              torch.ones(self.N_allC.shape).to(self.N)))

    def fold(self, tiles, agg):
        assert agg in ["sum", "mean"]
        x = torch.nn.Fold(output_size=self.last_input_size,
                          kernel_size=self.kernel_size,
                          stride=self.stride,
                          )(tiles.transpose(-2, -1))
        if agg == "mean":
            x = x / self.N
        elif agg == "sum":
            pass
        return x

    def forward(self, x, flag):
        self.init_params(x)
        tiles = self.unfold(x)

        # generate noise for every pixel in every tile
        u_0 = torch.randn(tiles.size()).to(tiles)
        u_mean = self.fold(u_0, agg="mean")
        u_centered = u_0 - self.unfold(u_mean)

        norm = self.unfold(self.fold(u_centered ** 2, agg="sum").sqrt())
        norm[:, self.unfold(self.N_allC).squeeze(0) == 1] = 1
        # this will denote the direction
        u_direction = u_centered / (norm+1e-6)
        # TODO is this necessary
        u_direction[:, self.unfold(self.N_allC).squeeze(0) == 1] = 0

        if self.augment == 'U':
            # sample U(0,A) for radius
            u_R = self.logA.exp() * self.unfold(torch.rand(x.size()).to(x) ** (1 / (self.N - 1)))
            L_aug = (self.unit_loglikelihoods - self.logA * (self.N - 1)).sum()
            L_aug = x.size(1) * L_aug

        if self.augment == 'N':
            # sample  radius
            chi2 = self.chi2.expand((x.size(0),)+self.N_allC.size()[1:])
            u_R = self.logA.exp() * chi2.rsample().sqrt().to(x)
            L_aug1 = (self.log_normal(u_R) * (self.N > 1)).flatten(1).sum(1)
            L_aug2 = x.size(1) * ((self.N-2)*(self.N > 1)).sum() * \
                self.log_normal(torch.tensor(0.))
            L_aug = L_aug1 + L_aug2
            u_R = self.unfold(u_R)

        u_final = u_R * u_direction

        # L_augment = (self.unit_loglikelihoods).sum() * x.size(1)
        L_scale = (self.N / 2 * self.N.log()).sum() * x.size(1)
        L = -L_aug + L_scale
        ######################################

        if flag == "sample":
            # apply noise
            tiles = tiles + self.unfold(self.N_allC.sqrt()) * u_final
        # reshape to (b,output_width,output_height,-1)
        num_X_patches = 1 + (x.size(2) - self.kernel_size[0]) // self.stride[0]
        num_Y_patches = 1 + (x.size(3) - self.kernel_size[1]) // self.stride[1]
        tiles = tiles.reshape(tiles.size(0), num_X_patches, num_Y_patches, -1)
        return tiles, L

    def inverse(self, tiles):
        tiles = tiles.view(tiles.size(0), -1, tiles.size(-1))
        x = self.fold(tiles, agg="mean")
        return x
