from nflows.distributions import StandardNormal
from nflows.flows import Flow
from nflows import transforms
import torch.nn as nn
from models.distributions.standard_normal import LogNormal
from torch.nn import functional as F


def coupling_inn(inp_dim, maker, nstack, tail_bound=3, tails='linear', lu=0, num_bins=10, mask=[1, 0],
                 unconditional_transform=False, spline=True, curtains_transformer=False):
    transform_list = []
    for i in range(nstack):
        # If a tail function is passed apply the same tail bound to every layer, if not then only use the tail bound on
        # the final layer
        tpass = tails
        if tails:
            tb = tail_bound
        else:
            tb = tail_bound if i == 0 else None
        if spline:
            transform_list += [
                transforms.PiecewiseRationalQuadraticCouplingTransform(mask, maker, tail_bound=tb, num_bins=num_bins,
                                                                       tails=tpass,
                                                                       apply_unconditional_transform=unconditional_transform)]
            if (tails == None) and (not tail_bound == None) and (i == nstack - 1):
                transform_list += [
                    transforms.standard.PointwiseAffineTransform(-tail_bound, 2 * tail_bound)]
        else:
            transform_list += [
                transforms.AffineCouplingTransform(mask, maker)]
            if unconditional_transform:
                warnings.warn(
                    'Currently the affine coupling layers only consider conditional transformations.')

        if lu:
            transform_list += [transforms.LULinear(inp_dim)]
        else:
            transform_list += [transforms.ReversePermutation(inp_dim)]

    if not (curtains_transformer and (nstack % 2 == 0)):
        # If the above conditions are satisfied then you want to permute back to the original ordering such that the
        # output features line up with their original ordering.
        transform_list = transform_list[:-1]

    return transforms.CompositeTransform(transform_list)


class dense_net(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        N = 256
        self.hidden_features = N
        self.N = nn.Sequential(
            nn.Linear(input_dim, N), nn.ReLU(),
            nn.Linear(N, N), nn.ReLU(),
            nn.Linear(N, N), nn.ReLU(),
            nn.Linear(N, output_dim))

    def forward(self, x, context):
        return self.N(x)


def maker(input_dim, output_dim):
    return dense_net(input_dim, output_dim)


class RQ_NSF(nn.Module):
    def __init__(self, dim, nstack):
        super().__init__()
        mask = dim // 2 * [0] + (dim - dim // 2) * [1]

        self.inn = coupling_inn(
            inp_dim=dim, nstack=nstack, maker=maker, num_bins=5, mask=mask)

    def forward(self, x, **kwargs):
        return self.inn.forward(x)

    def inverse(self, z, **kwargs):
        return self.inn.inverse(z)[0]


class conv_net(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        N = 32
        self.hidden_features = N
        self.N = nn.Sequential(
            nn.Conv2d(input_dim, N, padding=1, kernel_size=3), nn.ReLU(),
            nn.Conv2d(N, N, padding=1, kernel_size=3), nn.ReLU(),
            nn.Conv2d(N, N, padding=1, kernel_size=3), nn.ReLU(),
            nn.Conv2d(N, output_dim, padding=1, kernel_size=3)
        )

    def forward(self, x, context):
        return self.N(x)


def maker_conv(input_dim, output_dim):
    return conv_net(input_dim, output_dim)


class RQ_NSF_Conv(nn.Module):
    def __init__(self, channels, nstack):
        super().__init__()
        mask = channels // 2 * [0] + (channels - channels // 2) * [1]

        self.inn = coupling_inn(inp_dim=channels,
                                nstack=nstack,
                                maker=maker_conv,
                                num_bins=5,
                                mask=mask)

    def forward(self, x, **kwargs):
        return self.inn.forward(x)

    def inverse(self, z, **kwargs):
        return self.inn.inverse(z)[0]
