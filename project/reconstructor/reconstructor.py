import abc

import torch.nn as nn
from tensordict import TensorDict  # type: ignore


class SampleReconstructor(nn.Module):
    @abc.abstractmethod
    def forward(self, belief: TensorDict) -> TensorDict:
        """
        Reconstruct a sample from the belief distribution at a given accumulated precision. All parameters and the return
        Args:
            belief (TensorDict): The belief distribution.
            accumulated_precision (TensorDict): The accumulated precision of the belief.
        Returns:
            TensorDict: The reconstructed sample.
        """
        pass
