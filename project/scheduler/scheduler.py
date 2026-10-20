from abc import abstractmethod
from typing import Protocol

import torch


class Scheduler(Protocol):
    @abstractmethod
    def sample(self, t: torch.Tensor) -> torch.Tensor:
        """
        Get alpha(t) for the scheduler.
        """
