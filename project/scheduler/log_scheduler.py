import torch


class LogScheduler:
    def __init__(
        self,
        multiplier: float = 5.0,
        final_precision: float = 50.0,
        name: str = "LogScheduler",
    ):
        self.multiplier = multiplier
        self.final_precision = torch.tensor(final_precision)
        self.name = name

    def sample(self, t: torch.Tensor) -> torch.Tensor:
        """
        Get alpha(t) for the log scheduler.
        """
        const = torch.exp(-self.final_precision / self.multiplier)
        return torch.clamp(-self.multiplier * torch.log(1 - t + const), min=0.0)
