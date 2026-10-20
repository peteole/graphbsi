from abc import abstractmethod

import torch
from tensordict import TensorDict  # type: ignore
from torch import Tensor, softmax

from project.utils.nn import ScalarNyquistEmbedding
from project.utils.tensor_utils import expand_tensor


class GlobalFeaturesExtractor(torch.nn.Module):
    @abstractmethod
    def forward(self, belief: TensorDict) -> Tensor:
        """
        Extract global extra features from the belief distribution.

        Args:
            belief (TensorDict): The belief distribution.

        Returns:
            Tensor: The extracted global extra features. Shape is (*batch, extra_dims).
        """
        pass

    @property
    @abstractmethod
    def out_dim(self) -> int:
        """
        Get the output dimension of the global extra features.
        """
        pass


class TimeFeature(GlobalFeaturesExtractor):
    """
    Extracts time as a global extra feature.
    """

    def forward(self, belief: TensorDict) -> Tensor:
        """
        Extracts the time from the belief distribution.

        Args:
            belief (TensorDict): The belief distribution.

        Returns:
            Tensor: The time as a global extra feature. Shape is (*batch, 1).
        """
        return belief["t"].unsqueeze(-1)

    @property
    def out_dim(self) -> int:
        return 1


class TimeEmbeddingFeatures(GlobalFeaturesExtractor):
    """
    Extracts time as a global extra feature and embeds it using Nyquist embedding.
    """

    def __init__(self, size: int = 32, expected_rate: float = 100.0):
        super().__init__()
        self.embedding = ScalarNyquistEmbedding(size=size, expected_rate=expected_rate)

    def forward(self, belief: TensorDict) -> Tensor:
        """
        Extracts the time from the belief distribution and embeds it.

        Args:
            belief (TensorDict): The belief distribution.

        Returns:
            Tensor: The embedded time as a global extra feature. Shape is (*batch, embedding_dim).
        """
        t: Tensor = belief["t"]
        return self.embedding(t)

    @property
    def out_dim(self) -> int:
        return self.embedding.out_dim

class EntropyEmbeddingFeatures(GlobalFeaturesExtractor):
    """
    Extracts the entropy of the belief distribution as a global extra feature.
    """
    def __init__(self, size: int = 32, expected_rate: float = 100.0):
        super().__init__()
        self.embedding = ScalarNyquistEmbedding(size=size, expected_rate=expected_rate)

    def forward(self, belief: TensorDict) -> Tensor:
        """
        Extracts the entropy from the belief distribution.

        Args:
            belief (TensorDict): The belief distribution.

        Returns:
            Tensor: The entropy as a global extra feature. Shape is (*batch, 1).
        """
        node_probs = softmax(belief["X"], dim=-1)
        edge_probs = softmax(belief["E"], dim=-1)
        node_entropy = -torch.sum(node_probs * torch.log(node_probs + 1e-10), dim=-1)
        node_mask = belief["mask"]
        node_entropy = node_entropy * expand_tensor(node_mask,node_entropy.shape)
        edge_entropy = -torch.sum(edge_probs * torch.log(edge_probs + 1e-10), dim=-1)
        edge_mask = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
        edge_entropy = edge_entropy * expand_tensor(edge_mask,edge_entropy.shape)
        entropy = node_entropy.mean(dim=tuple(range(1, node_entropy.ndim))) + edge_entropy.mean(dim=tuple(range(1, edge_entropy.ndim)))
        return self.embedding(entropy)

    @property
    def out_dim(self) -> int:
        return self.embedding.out_dim