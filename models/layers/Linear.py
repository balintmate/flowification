import math

import torch
import torch.nn as nn
import wandb

from models.layers.base import Layer
from models.layers.funnel import Funnel



class Rotation(Layer):

    def __init__(self, dimension):
        super(Rotation, self).__init__()
        self.a_mat = nn.Parameter(torch.randn(dimension, dimension))

    @property
    def rotation(self):
        return torch.matrix_exp(self.log_u_mat)

    @property
    def inverse_rotation(self):
        return torch.matrix_exp(-self.log_u_mat)

    @property
    def log_u_mat(self):
        return self.a_mat - self.a_mat.T

    def forward(self, x, **kwargs):
        return x @ self.rotation

    def inverse(self, z):
        return z @ self.inverse_rotation


class DimMatch(Layer):
    def __init__(self, dimension):
        super(DimMatch, self).__init__()
        self.dimension = dimension
        self.logW = nn.Parameter(
            torch.zeros(1, dimension)/math.sqrt(dimension)
        )

    def forward(self, x, **kwargs):
        return self.logW.exp() * x, self.logW.sum(1).repeat(len(x))

    def inverse(self, z, **kwargs):
        return (-self.logW).exp() * z[:, :self.dimension]


class DimIncrease(Layer):
    def __init__(self, D1, D2):
        super(DimIncrease, self).__init__()
        self.D1 = D1
        self.D2 = D2
        self.logW = nn.Parameter(torch.randn(1, D2)/math.sqrt(D2))

    def forward(self, x, flag):
        padding = torch.rand(x.size(0), self.D2 - self.D1).to(x) - .5
        if flag == 'mean':
            padding = 0 * padding
        x = torch.cat((x, padding), -1)
        z = self.logW.exp() * x
        return z, self.logW.sum(1).repeat(len(z))

    def inverse(self, z, **kwargs):
        return (-self.logW[:, :self.D1]).exp() * z[:, :self.D1]


class Linear(nn.Module):
    def __init__(self, D1, D2):
        super().__init__()
        self.D1 = D1
        self.D2 = D2
        self.U1 = Rotation(D1)
        self.U2 = Rotation(D2)

        # bias of the layer
        self.b = nn.Parameter(torch.zeros(1, D2))

        if D1 > D2:
            self.funnel = Funnel(D1, D2)
        elif D1 == D2:
            self.funnel = DimMatch(D1)
        elif D2 > D1:
            self.funnel = DimIncrease(D1, D2)

    @property
    def weight_matrix(self):
        if self.D2 > self.D1:
            raise Exception(
                'There is no equivalent dimension increasing MLP without noise injection.')
        funnel = self.funnel.logW.exp().squeeze().diag()
        if self.D1 >= self.D2:
            zeros = torch.zeros((self.D2, self.D1 - self.D2)).to(funnel)
            funnel = torch.cat((zeros, funnel), 1)
        return self.U2.rotation.transpose(0, 1) @ funnel @ self.U1.rotation.transpose(0, 1)

    def forward(self, x, flag):
        in_size = x.size()
        x = x.reshape(-1, self.D1)
        x = self.U1(x)
        z, likelihood_contribution = self.funnel(x, flag=flag)
        z = self.U2(z)
        z = z + self.b
        return z.view(in_size[:-1] + (self.D2,)), likelihood_contribution.view(in_size[:-1])

    def inverse(self, z, flag):
        in_size = z.size()
        z = z.reshape(-1, self.D2)
        z = z - self.b
        z = self.U2.inverse(z)
        x = self.funnel.inverse(z, flag=flag)
        x = self.U1.inverse(x)
        return x.view(in_size[:-1] + (self.D1,))
