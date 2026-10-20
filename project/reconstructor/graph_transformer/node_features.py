from abc import abstractmethod

import torch


class NodeFeaturesExtractor(torch.nn.Module):
    """
    Base class for node features extractors.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, belief) -> torch.Tensor:
        """
        Extracts node features from the belief state.

        Args:
            belief (TensorDict): The belief state containing node features.

        Returns:
            Tensor: The extracted node features. Shape is (*batch, n, node_dims).
        """
        pass

    @property
    @abstractmethod
    def out_dim(self) -> int:
        """
        Get the output dimension of the node features.
        """
        pass


class ClassProbabilityNodeFeatures(NodeFeaturesExtractor):
    """
    Extracts node features as class probabilities.
    """

    def __init__(self, num_classes: int):
        """
        Initializes the ClassProbabilityNodeFeatures extractor.

        Args:
            num_classes (int): The number of classes for node features.
        """
        super().__init__()
        self.num_classes = num_classes

    def forward(self, belief) -> torch.Tensor:
        return torch.softmax(belief["X"], dim=-1)

    @property
    def out_dim(self) -> int:
        return self.num_classes


class ClassLogitsNodeFeatures(NodeFeaturesExtractor):
    """
    Extracts node features as class logits.
    """

    def __init__(self, num_classes: int, normalize: bool = True):
        """
        Initializes the ClassLogitsNodeFeatures extractor.

        Args:
            num_classes (int): The number of classes for node features.
        """
        super().__init__()
        self.num_classes = num_classes
        self.normalize = normalize

    def forward(self, belief) -> torch.Tensor:
        if self.normalize:
            return belief["X"] - belief["X"].mean(dim=-1, keepdim=True)
        return belief["X"]

    @property
    def out_dim(self) -> int:
        return self.num_classes


class EntropyNodeFeatures(NodeFeaturesExtractor):
    """
    Extracts node features as entropy of the class probabilities.
    """

    def forward(self, belief) -> torch.Tensor:
        """
        Extracts the entropy of the class probabilities from the belief state.

        Args:
            belief (TensorDict): The belief state containing node features.

        Returns:
            Tensor: The entropy of the class probabilities. Shape is (*batch, n, 1).
        """
        node_probs = torch.softmax(belief["X"], dim=-1)
        return -torch.sum(node_probs * torch.log(node_probs + 1e-10), dim=-1).unsqueeze(-1)

    @property
    def out_dim(self) -> int:
        return 1


class RRWPNodeFeatures(NodeFeaturesExtractor):
    """
    Extracts the RRWP (Relative Random Walk Probability) as a global extra feature.
    """

    def __init__(self, max_steps: int = 12, make_symmetric: bool = False):
        super().__init__()
        self.max_steps = max_steps
        self.make_symmetric = make_symmetric

    def forward(self, belief):
        E_class_probs = torch.softmax(belief["E"], dim=-1)
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

        rrwp_features = torch.stack(rrwp_list, -1)  # shape is (*batch, n, n, max_steps)
        diag = rrwp_features.diagonal(dim1=-2, dim2=-1)  # shape is (*batch, n, max_steps)
        return diag

    @property
    def out_dim(self) -> int:
        return self.max_steps

class DegreeNodeFeatures(NodeFeaturesExtractor):
    """
    Extracts node features as the degree of each node.
    """

    def forward(self, belief) -> torch.Tensor:
        E_class_probs = torch.softmax(belief["E"], dim=-1)
        E_class_probs = (E_class_probs + E_class_probs.transpose(1, 2)) / 2  # make symmetric
        E = 1 - E_class_probs[..., 0]  # First class is the no edge class. Shape is (*batch, n, n)
        degrees = E.sum(dim=-1).unsqueeze(-1)  # shape is (*batch, n, 1)
        return degrees

    @property
    def out_dim(self) -> int:
        return 1
