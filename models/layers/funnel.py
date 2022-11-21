from audioop import mul
import torch
import torch.nn as nn
import math
import wandb


class SVD(nn.Module):
    def __init__(self, D1, D2):
        super().__init__()
        self.D1 = D1
        self.D2 = D2
        assert D1 > D2

        # diagonal scaling in the target space
        self.logW = nn.Parameter(torch.randn(1, D2)/math.sqrt(D2))
        self.X1_inv = Gaussian_X1(D1, D2)

    def forward(self, x):
        x1, x2 = torch.split(x, [self.D1 - self.D2, self.D2], 1)
        z = self.logW.exp() * x2
        return z, torch.zeros(x.shape[0]).to(z)

    def inverse(self, z, flag):
        x1 = self.X1_inv.inverse(z, flag=flag)
        x2 = (-self.logW).exp() * z
        x = torch.cat((x1, x2), 1)
        return x


class Funnel(SVD):

    def forward(self, x, **kwargs):
        x1, x2 = torch.split(x, [self.D1 - self.D2, self.D2], 1)
        z = self.logW.exp() * x2
        logpx1 = self.X1_inv.loglikelihood(x1, z)
        return z, self.logW.sum() + logpx1


class Gaussian_X1(nn.Module):
    def __init__(self, D1, D2):
        super().__init__()
        self.D1 = D1
        self.D2 = D2
        N = max(3 * max(2 * (D1 - D2), D1), 64)
        if N > 200:
            numhiddenL = 0
        elif N > 100:
            numhiddenL = 2
        elif N > 50:
            numhiddenL = 4
        else:
            numhiddenL = 6
        layers = [nn.Linear(D2, N), nn.ReLU()]
        for _ in range(numhiddenL):
            layers += [nn.Linear(N, N), nn.ReLU()]
        layers += [nn.Linear(N, 2 * (D1 - D2))]
        self.p = nn.Sequential(*layers)

    def inverse(self, z, flag):
        mu, logstd = torch.split(self.p(z), self.D1 - self.D2, 1)
        if flag == 'mean':
            x1 = mu
        elif flag == 'sample':
            x1 = mu + torch.randn(logstd.size()).to(z.device) * logstd.exp()
        return x1

    def loglikelihood(self, x1, z):
        mu, logstd = torch.split(self.p(z), self.D1 - self.D2, 1)
        std = logstd.exp()
        exp = -.5 * (((mu - x1) / std) ** 2)
        Z = -.5 * math.log(2 * math.pi) - logstd
        logpx1 = exp + Z
        return logpx1.sum(1)
