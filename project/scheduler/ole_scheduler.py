from math import log

import torch


class OLEScheduler:
    def __init__(
        self,
        initial_precision: float = 1,
        final_precision: float = 50.0,
        name: str = "OLEScheduler",
    ):
        self.initial_precision = initial_precision
        self.final_precision = final_precision
        self.name = name

    def sample(self, t: torch.Tensor) -> torch.Tensor:
        """
        Get alpha(t) for the OLE scheduler.
        """
        return (
            self.initial_precision
            * torch.exp(t * log(self.final_precision / self.initial_precision))
            - self.initial_precision
        )
