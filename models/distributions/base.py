import nflows.distributions
import torch


class Distribution(nflows.distributions.Distribution):
    def __init__(self, dimension, dtype=torch.float32):
        super(Distribution, self).__init__()
        self.dimension = dimension
        self.register_buffer('device', torch.zeros(1, dtype=dtype))
