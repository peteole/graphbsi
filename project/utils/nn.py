
import numpy as np
import torch
from torch import Tensor


class ScalarNyquistEmbedding(torch.nn.Module):
    """
    A module that embeds scalar values using Nyquist-frequency-based sinusoidal embeddings.

    This creates a positional encoding that uses geometrically spaced frequencies
    up to the Nyquist frequency, with sine and cosine components for each frequency.
    """

    def __init__(self, size: int = 32, expected_rate: float = 100.0):
        """
        Initialize the scalar Nyquist embedding.

        Args:
            size (int): The embedding dimension. Must be even.
            expected_rate (float): The expected sampling rate per unit interval.
        """
        super().__init__()
        self.size = size

        assert size % 2 == 0, "Embedding size must be even"

        k = size // 2

        # Nyquist frequency for a given sampling rate per unit interval
        nyquist_frequency = expected_rate / 2

        golden_ratio = (1 + np.sqrt(5)) / 2
        frequencies = np.geomspace(1 / 8, nyquist_frequency / (2 * golden_ratio), num=k)

        # Sample every frequency twice, once shifted by pi/2 to get cosine
        scale = np.repeat(2 * np.pi * frequencies, 2)
        bias = np.tile(np.array([0, np.pi / 2]), k)

        self.register_buffer(
            "scale", torch.tensor(scale, dtype=torch.float32), persistent=False
        )
        self.register_buffer(
            "bias", torch.tensor(bias, dtype=torch.float32), persistent=False
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Embed scalar values using Nyquist-frequency-based sinusoidal embeddings.

        Args:
            x (Tensor): Input scalar values. Shape is (*batch,).

        Returns:
            Tensor: Embedded values. Shape is (*batch, embedding_dim).
        """
        return torch.addcmul(self.bias, self.scale, x[..., None]).sin()  # type: ignore

    @property
    def out_dim(self) -> int:
        """Get the output dimension of the embedding."""
        return self.size