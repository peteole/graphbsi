from abc import abstractmethod

import torch
from tensordict import TensorDict
from torch import Tensor


class ClassConditioner:
    """
    A conditioner for a case where each ground truth sample has a class that should be fed to the reconstructor during training.
    During inference, the class is sampled from the distribution of classes in the dataset.
    """

    @abstractmethod
    def classify_sample(self, sample: TensorDict) -> Tensor:
        """
        Classify a batch of samples.
        Args:
            sample (TensorDict): The batch of samples to classify.
        Returns:
            Tensor: The classification of the sample.
        """
        ...

    @abstractmethod
    def write_sample_classifications_to_belief(
        self, classifications: Tensor, belief: TensorDict
    ) -> None:
        """
        Write a batch of sample classifications to the belief.
        Args:
            classifications (Tensor): The batch of classifications of the samples.
            belief (TensorDict): The belief to write the classifications to.
        """
        ...

    @abstractmethod
    def sample_classifications(
        self, num_samples: int, device: torch.device | None = None
    ) -> Tensor:
        """
        Sample classifications from the distribution of classes in the dataset.
        Args:
            num_samples (int): The number of samples to sample.
        Returns:
            Tensor: The sampled classifications.
        """
        ...
