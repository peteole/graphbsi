from math import sqrt

import torch
from torch.distributions import Normal

from project.prior.prior import Prior


class GaussianPrior(Prior):
    """
    A belief that is a categorical distribution.
    """

    @classmethod
    def from_config(
        cls,
        shape: tuple,
        device: str | torch.device = "cpu",
        initial_precision: float = 1.0,
    ):
        return cls(
            shape=shape,
            device=device,
            initial_precision=initial_precision,
        )

    def __init__(
        self,
        shape: tuple,
        device: str | torch.device = "cpu",
        initial_precision: float = 1.0,
    ):
        super().__init__()
        self.distribution = Normal(
            torch.zeros(tuple(shape), device=device),
            torch.ones(tuple(shape), device=device) / sqrt(initial_precision),
        )
        self.shape = shape
        self.device = torch.device(device)
        self.initial_precision = initial_precision

    def sample(self, batch_size):
        if isinstance(batch_size, int):
            batch_size = (batch_size,)
        return self.distribution.sample(batch_size)

    def to(self, device: str | torch.device):
        self.distribution.loc = self.distribution.loc.to(device)
        self.distribution.scale = self.distribution.scale.to(device)
        self.device = torch.device(device)
