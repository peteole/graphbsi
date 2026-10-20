"""
Sequence reconstructor for DNA sequences using CNN.
"""

import torch
from tensordict import TensorDict

from project.data.sequence import SequenceDataModule
from project.reconstructor.reconstructor import SampleReconstructor
from project.reconstructor.sequence import CNNModel, CNNModelConfig


def expand_simplex(
    xt: torch.Tensor, alphas: torch.Tensor, prior_pseudocount: float = 2.0
):
    """
    Expand simplex representation for CNN input.

    This function doubles the input channels by weighting the input with
    and without a prior weight. This is used in the dirichlet-flow-matching
    paper for better representation learning.

    Args:
        xt: Input tensor of shape (B, L, A) where B is batch size,
            L is sequence length, A is alphabet size
        alphas: Alpha values of shape (B,) controlling the prior weight
        prior_pseudocount: Prior pseudocount hyperparameter (default: 2.0)

    Returns:
        Tuple of:
            - Expanded tensor of shape (B, L, A*2)
            - Prior weights of shape (B, 1, 1)
    """
    prior_weights = (prior_pseudocount / (alphas + prior_pseudocount - 1))[:, None, None]
    return torch.cat([xt * (1 - prior_weights), xt * prior_weights], -1), prior_weights


class SequenceReconstructor(SampleReconstructor):
    """
    A reconstructor for DNA sequences using a CNN model.
    Takes a belief (categorical distribution over nucleotides) and reconstructs
    the sequence distribution.
    """

    @classmethod
    def from_config(
        cls,
        data_module: SequenceDataModule,
        num_cnn_stacks: int = 4,
        hidden_dim: int = 128,
        dropout: float = 0.0,
        **kwargs,
    ) -> "SequenceReconstructor":
        """
        Create a SequenceReconstructor from configuration and data module.

        Args:
            data_module: The data module providing dataset information
            num_cnn_stacks: Number of CNN stacks (default: 4)
            hidden_dim: Hidden dimension for the CNN (default: 128)
            dropout: Dropout rate (default: 0.0)
            **kwargs: Additional arguments (ignored for compatibility)

        Returns:
            SequenceReconstructor instance
        """
        # Ensure data module is set up
        if not hasattr(data_module, "num_cls") or data_module.num_cls == 0:
            data_module.prepare_data()
            data_module.setup("fit")

        return cls(
            alphabet_size=data_module.alphabet_size,
            num_cls=data_module.num_cls,
            sequence_length=data_module.sequence_length,
            num_cnn_stacks=num_cnn_stacks,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    def __init__(
        self,
        alphabet_size: int,
        num_cls: int,
        sequence_length: int,
        num_cnn_stacks: int = 4,
        hidden_dim: int = 128,
        dropout: float = 0.0,
    ):
        """
        Initialize the SequenceReconstructor.

        Args:
            alphabet_size: Number of nucleotides (4 for A, C, G, T)
            num_cls: Number of classes for conditional generation
            sequence_length: Length of the sequences
            num_cnn_stacks: Number of CNN stacks
            hidden_dim: Hidden dimension for the CNN
            dropout: Dropout rate
        """
        super().__init__()

        self.alphabet_size = alphabet_size
        self.num_cls = num_cls
        self.sequence_length = sequence_length

        # Create CNN config
        config = CNNModelConfig(
            hidden_dim=hidden_dim,
            num_cnn_stacks=num_cnn_stacks,
            dropout=dropout,
            clean_data=False,
            cls_expanded_simplex=False,
            mode="dirichlet",
            cls_free_guidance=False,
        )

        # Create the CNN model (not a classifier)
        self.cnn = CNNModel(
            config=config,
            alphabet_size=alphabet_size,
            num_cls=num_cls,
            classifier=False,
        )

    def forward(self, belief: TensorDict) -> TensorDict:
        """
        Reconstruct sequence distribution from belief.

        Args:
            belief: TensorDict containing:
                - 'sequence': (batch_size, sequence_length, alphabet_size) - belief over nucleotides
                - 't': (batch_size,) - normalized precision values

        Returns:
            TensorDict containing:
                - 'sequence': (batch_size, sequence_length, alphabet_size) - reconstructed logits
        """
        # Extract belief components
        sequence_belief = belief["sequence"]  # (B, L, A)
        t = belief["t"]  # (B,)

        batch_size = sequence_belief.shape[0]

        # If the CNN was configured with expanded_simplex_input=True,
        # we need to expand the input from A channels to 2*A channels
        # Check if expansion is needed by looking at the CNN's expected input size
        if self.cnn.config.cls_expanded_simplex or (
            not self.cnn.classifier
            and (
                self.cnn.config.mode == "dirichlet"
                or self.cnn.config.mode == "riemannian"
            )
        ):
            # Expand simplex: double the channels using prior weighting
            # Convert precision to alpha: higher precision -> higher alpha
            # Alpha should be >= 1, so we use alpha = t + 1
            alphas = t + 1.0  # (B,)
            sequence_belief_expanded, _ = expand_simplex(
                sequence_belief, alphas, prior_pseudocount=2.0
            )
            input_to_cnn = sequence_belief_expanded  # (B, L, 2*A)
        else:
            input_to_cnn = sequence_belief  # (B, L, A)

        # The CNN expects input with shape (B, L, A) or (B, L, 2*A)
        # It will internally permute to (B, A, L) or (B, 2*A, L) for conv operations
        cnn_output = self.cnn(input_to_cnn, t, cls=None)  # (B, L, A)
        logits = cnn_output + sequence_belief # (B, L, A), add skip connection preconditioning
        normalized = torch.softmax(logits, dim=-1)

        # Return reconstructed sequence logits
        return TensorDict(
            {
                "sequence": normalized,
            },
            batch_size=batch_size,
        )

    def __repr__(self):
        return (
            f"SequenceReconstructor(\n"
            f"  alphabet_size={self.alphabet_size},\n"
            f"  num_cls={self.num_cls},\n"
            f"  sequence_length={self.sequence_length},\n"
            f"  num_layers={self.cnn.num_layers}\n"
            f")"
        )
