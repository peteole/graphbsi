import torch
from torch.distributions import Dirichlet

from project.prior.prior import Prior


class DirichletPrior(Prior):
    """
    Prior for a categorical belief where the belief is sampled from a Dirichlet distribution.
    """

    @classmethod
    def from_config(
        cls,
        shape: tuple,
        num_classes: int,
        concentration: float = 1.0,
        device: str | torch.device = "cpu",
    ):
        return cls(
            shape=shape,
            num_classes=num_classes,
            concentration=concentration,
            device=device,
        )

    def __init__(
        self,
        shape: tuple,
        num_classes: int,
        concentration: float = 1.0,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        self.num_classes = num_classes
        self.device = torch.device(device)

        self.distribution = Dirichlet(
            torch.full(
                tuple(shape) + (num_classes,),
                concentration,
                device=self.device if self.device.type != "mps" else None,
            )
        )
        self.shape = shape

    def sample(self, batch_size):
        if isinstance(batch_size, int):
            batch_size = (batch_size,)
        samples = self.distribution.sample(batch_size)
        if self.device.type == "mps":  # Dirichlet sampling is not supported on MPS yet
            samples = samples.to(self.device)
        return samples.log()

    def to(self, device: str | torch.device):
        self.device = torch.device(device)
        if self.device.type != "mps":
            self.distribution.concentration = self.distribution.concentration.to(
                self.device
            )
