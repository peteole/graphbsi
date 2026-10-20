from abc import abstractmethod
from typing import Protocol

from tensordict import TensorDict


class Conditioner(Protocol):
    @abstractmethod
    def __call__(self, reconstruction: TensorDict) -> None:
        """
        Modify a batch of reconstructions in place to condition the generation.
        Args:
            reconstruction: A batch of reconstructions.
        """
        pass
