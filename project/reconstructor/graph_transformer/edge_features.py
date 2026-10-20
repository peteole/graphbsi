from abc import abstractmethod

import torch
from tensordict import TensorDict  # type: ignore
from torch import Tensor, softmax


class EdgeFeaturesExtractor(torch.nn.Module):
    @abstractmethod
    def forward(self, belief: TensorDict) -> Tensor:
        """
        Extract edge features from the belief distribution.

        Args:
            belief (TensorDict): The belief distribution.

        Returns:
            Tensor: The extracted edge features. Shape is (*batch, n, n, edge_dims).
        """
        pass

    @property
    @abstractmethod
    def out_dim(self) -> int:
        """
        Get the output dimension of the edge features.
        """
        pass


class ClassProbabilityEdgeFeatures(EdgeFeaturesExtractor):
    """
    Extracts edge features as class probabilities.
    """

    def __init__(self, num_classes: int):
        """
        Initializes the ClassProbabilityEdgeFeatures extractor.

        Args:
            num_classes (int): The number of classes for edge features.
        """
        super().__init__()
        self.num_classes = num_classes

    def forward(self, belief: TensorDict) -> Tensor:
        return softmax(belief["E"], dim=-1)

    @property
    def out_dim(self) -> int:
        return self.num_classes


class ClassLogitsEdgeFeatures(EdgeFeaturesExtractor):
    """
    Extracts edge features as class logits
    """

    def __init__(self, num_classes: int, normalize: bool = True):
        """
        Initializes the ClassLogitsEdgeFeatures extractor.

        Args:
            num_classes (int): The number of classes for edge features.
        """
        super().__init__()
        self.num_classes = num_classes
        self.normalize = normalize

    def forward(self, belief: TensorDict) -> Tensor:
        if self.normalize:
            return belief["E"] - belief["E"].mean(dim=-1, keepdim=True)
        return belief["E"]

    @property
    def out_dim(self) -> int:
        return self.num_classes


class RRWPEdgeFeatures(EdgeFeaturesExtractor):
    """
    Extracts the RRWP (Relative Random Walk Probability) as a global extra feature.
    """

    def __init__(self, max_steps: int = 12, make_symmetric: bool = False):
        super().__init__()
        self.max_steps = max_steps
        self.make_symmetric = make_symmetric

    def forward(self, belief: TensorDict) -> Tensor:
        E_class_probs = softmax(belief["E"], dim=-1)
        if self.make_symmetric:
            # Make the edge probabilities symmetric
            E_class_probs = (E_class_probs + E_class_probs.transpose(1, 2)) / 2
        E = (
            1 - E_class_probs[..., 0]
        )  # First class is the no edge class, so we take 1 - E_class_probs[..., 0] to get the edge probabilities.
        (
            bs,
            n,
            n2,
        ) = E.shape
        assert n == n2, "Edge features must be square matrix."

        # Create transition matrix: normalize each row to sum to 1
        row_sums = E.sum(dim=-1, keepdim=True).float()
        # Add small epsilon to avoid division by zero
        row_sums = torch.clamp(row_sums, min=1e-10)
        transition_matrix = E / row_sums

        id = torch.eye(n, device=E.device).unsqueeze(0).repeat(bs, 1, 1)
        rrwp_list = [id]

        for i in range(self.max_steps - 1):
            cur_rrwp = rrwp_list[-1] @ transition_matrix
            rrwp_list.append(cur_rrwp)

        return torch.stack(rrwp_list, -1)  # shape is (*batch, n, n, max_steps)

    @property
    def out_dim(self) -> int:
        return self.max_steps


class EntropyEdgeFeatures(EdgeFeaturesExtractor):
    """
    Extracts the entropy of the edge features as a  feature.
    """

    def __init__(self):
        """
        Initializes the EntropyEdgeFeatures extractor.
        """
        super().__init__()

    def forward(self, belief: TensorDict) -> Tensor:
        probabilities = softmax(belief["E"], dim=-1)
        return -torch.sum(
            probabilities * torch.log(probabilities + 1e-10), dim=-1
        ).unsqueeze(-1)

    @property
    def out_dim(self) -> int:
        return 1  # Entropy is a single value per edge
