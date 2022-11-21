import wandb

from models.distributions.base import Distribution


class Flow(Distribution):

    def __init__(self, model, base_distribution):
        super(Flow, self).__init__(base_distribution.dimension)
        self.model = model
        self.base_distribution = base_distribution

    def sample(self, flag, **kwargs):
        if 'z' in kwargs:
            z = kwargs['z']
        else:
            z = self.base_distribution.sample(nsamples)
        return self.inverse(z, flag=flag)

    def forward(self, x, flag='sample'):
        x, loglikelihood = self.model(x, flag=flag)
        log_pz = self.base_distribution.log_prob(x)
        if flag != 'mean':
            wandb.log({'log p(z)': log_pz.mean().item()})
        loglikelihood += log_pz
        return x, loglikelihood

    def inverse(self, z, flag='mean'):
        return self.model.inverse(z, flag=flag)

    def log_prob(self, x, context=None):
        return self(x)[1]
