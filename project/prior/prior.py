from abc import abstractmethod

import torch


class Prior:
    def __init__(self):
        pass

    @abstractmethod
    def sample(self, batch_size: int | tuple | torch.Size) -> torch.Tensor:
        """
        Get the prior distribution.
        """
        ...

    def to(self, device: str | torch.device):
        """
        Move the prior to a specific device.
        """
        pass
