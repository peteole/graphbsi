import torch


class LinScheduler:
    def __init__(self, final_precision: float = 1, name: str = "LinScheduler"):
        self.final_precision = final_precision
        self.name = name

    def sample(self, t: torch.Tensor) -> torch.Tensor:
        """
        Get alpha(t) for the linear scheduler.
        """
        return self.final_precision * t
