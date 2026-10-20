import torch
from torch.distributions import (
    Normal,
)

from project.prior.prior import Prior


class SoftmaxNormalPrior(Prior):
    """
    A belief that is a categorical distribution.
    """

    def __init__(
        self,
        shape: tuple,
        num_classes: int,
        device: str | torch.device = "cpu",
        initial_belief_std: float = 0.2,
        mean_probabilities: list[float] | torch.Tensor | None = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        if mean_probabilities is None:
            mean_probabilities = [1.0 / num_classes] * num_classes
        mean_probabilities = torch.tensor(
            mean_probabilities, device=device, dtype=torch.float32
        )
        assert mean_probabilities.shape == (
            num_classes,
        ) or mean_probabilities.shape == shape + (num_classes,), (
            f"mean_probabilities must be of shape ({num_classes},) or {shape + (num_classes,)} but got {mean_probabilities.shape}"
        )
        if mean_probabilities.shape == (num_classes,):
            mean_probabilities = mean_probabilities.view(
                (1,) * len(shape) + (num_classes,)
            ).expand(tuple(shape) + (num_classes,))
        assert mean_probabilities.shape == shape + (num_classes,), (
            f"mean_probabilities must be of shape {shape + (num_classes,)} but got {mean_probabilities.shape}"
        )
        mean_probabilities = mean_probabilities + 5e-3
        mean_probabilities = mean_probabilities / mean_probabilities.sum(dim=-1, keepdim=True)
        mean_probabilities = mean_probabilities.log()  # The softmax of the belief is the probability so to get the probabilities as prior we need to log them
        mean_probabilities = mean_probabilities - mean_probabilities.mean(
            dim=-1, keepdim=True
        ) # Normalize the probabilities
        mean_probabilities = mean_probabilities.to(device)
        self.distribution = Normal(
            loc=mean_probabilities,
            scale=torch.full(tuple(shape) + (num_classes,), initial_belief_std, device=device),
        )
        # self.distribution = TransformedDistribution(
        #     self.initial_base_dist, SoftmaxTransform()
        # )
        self.shape = shape
        self.device = device

    def sample(self, batch_size):
        if isinstance(batch_size, int):
            batch_size = (batch_size,)
        sample: torch.Tensor = self.distribution.sample(batch_size)  # type: ignore
        return sample

    def to(self, device: str | torch.device):
        self.distribution.loc = self.distribution.loc.to(device)
        self.distribution.scale = self.distribution.scale.to(device)
