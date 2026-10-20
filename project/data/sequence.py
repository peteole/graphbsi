from pathlib import Path
from typing import Literal, TypedDict
import torch
import pickle
import copy
import lightning.pytorch as pl
from torch.utils.data import DataLoader, Dataset
from molecule_benchmarks.utils import available_cpu_count


class SequenceSample(TypedDict):
    """A typed dictionary representing a sequence sample."""

    sequence: torch.Tensor  # DNA sequence as integer-encoded tensor
    mask: torch.BoolTensor  # Mask indicating valid positions in the sequence
    cls: torch.Tensor  # Class label


class EnhancerDataset(Dataset):
    """Internal dataset class for enhancer sequences."""

    def __init__(
        self,
        seqs: torch.Tensor,
        clss: torch.Tensor,
        alphabet_size: int,
        num_cls: int,
    ):
        self.seqs = seqs
        self.clss = clss
        self.alphabet_size = alphabet_size
        self.num_cls = num_cls

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx) -> SequenceSample:
        seq = self.seqs[idx]
        # Create mask (all positions are valid for these fixed-length sequences)
        mask: torch.BoolTensor = torch.ones_like(seq, dtype=torch.bool)  # type: ignore
        return {
            "sequence": seq,
            "mask": mask,
            "cls": self.clss[idx],
        }


class SequenceDataModule(pl.LightningDataModule):
    """
    A Lightning DataModule for handling DNA sequence data (e.g., enhancers).
    """

    @classmethod
    def from_config(
        cls,
        cache_dir: str | Path,
        ds_type: Literal["FlyBrain", "MEL2"],
        batch_size: int = 32,
        num_workers: int = 4,
        sequence_length: int = 500,
        **kwargs,
    ) -> "SequenceDataModule":
        """
        Create a SequenceDataModule from configuration parameters.

        Args:
            cache_dir: Directory containing the dataset files
            ds_type: Type of dataset ("FlyBrain" or "MEL2")
            batch_size: Batch size for dataloaders
            num_workers: Number of workers for dataloaders
            **kwargs: Additional arguments (ignored for compatibility)

        Returns:
            SequenceDataModule instance
        """
        return cls(
            data_dir=Path(cache_dir),
            version=ds_type,
            batch_size=batch_size,
            num_workers=num_workers,
            seq_length=sequence_length,
        )

    def __init__(
        self,
        data_dir: str | Path,
        version: Literal["MEL2", "FlyBrain"],
        batch_size: int = 32,
        num_workers: int = 4,
        seq_length: int = 500,
    ):
        """
        Initialize the SequenceDataModule.

        Args:
            data_dir: Directory containing the dataset files
            version: Dataset version ("MEL2" or "FlyBrain")
            batch_size: Batch size for dataloaders
            num_workers: Number of workers for dataloaders
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self.version = version
        self.batch_size = batch_size
        self.num_workers = min(num_workers, available_cpu_count())

        # Metadata
        self.alphabet_size = 4  # A, C, G, T
        self.num_cls: int = 0
        self.sequence_length = seq_length

        # Datasets
        self.train_dataset: EnhancerDataset | None = None
        self.val_dataset: EnhancerDataset | None = None
        self.test_dataset: EnhancerDataset | None = None

    def prepare_data(self):
        """
        Download or verify data exists. Called only on one GPU/process.
        """
        data_file = self.data_dir / f"Deep{self.version}_data.pkl"
        if not data_file.exists():
            raise FileNotFoundError(
                f"Dataset file not found: {data_file}. "
                f"Please ensure the Deep{self.version}_data.pkl file exists in {self.data_dir}"
            )

    def setup(self, stage: str | None = None):
        """
        Load data and create datasets. Called on every GPU/process.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'
        """
        data_file = self.data_dir / f"Deep{self.version}_data.pkl"
        all_data = pickle.load(open(data_file, "rb"))

        # Extract metadata
        self.num_cls = all_data["y_train"].shape[-1]

        if stage == "fit" or stage is None:
            # Load training data
            train_seqs = torch.argmax(
                torch.from_numpy(copy.deepcopy(all_data["train_data"])), dim=-1
            )
            train_clss = torch.argmax(
                torch.from_numpy(copy.deepcopy(all_data["y_train"])), dim=-1
            )
            # self.sequence_length = train_seqs.shape[-1]

            self.train_dataset = EnhancerDataset(
                seqs=train_seqs,
                clss=train_clss,
                alphabet_size=self.alphabet_size,
                num_cls=self.num_cls,
            )

            # Use test split as validation (following the original structure)
            val_seqs = torch.argmax(
                torch.from_numpy(copy.deepcopy(all_data["test_data"])), dim=-1
            )
            val_clss = torch.argmax(
                torch.from_numpy(copy.deepcopy(all_data["y_test"])), dim=-1
            )

            self.val_dataset = EnhancerDataset(
                seqs=val_seqs,
                clss=val_clss,
                alphabet_size=self.alphabet_size,
                num_cls=self.num_cls,
            )

        if stage == "test" or stage is None:
            # Load test data
            test_seqs = torch.argmax(
                torch.from_numpy(copy.deepcopy(all_data["test_data"])), dim=-1
            )
            test_clss = torch.argmax(
                torch.from_numpy(copy.deepcopy(all_data["y_test"])), dim=-1
            )

            self.test_dataset = EnhancerDataset(
                seqs=test_seqs,
                clss=test_clss,
                alphabet_size=self.alphabet_size,
                num_cls=self.num_cls,
            )

    def train_dataloader(self) -> DataLoader:
        """Return the training dataloader."""
        assert self.train_dataset is not None, (
            "Train dataset not initialized. Call setup() first."
        )
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=True,
            prefetch_factor=2 if self.num_workers > 0 else None,
        )

    def val_dataloader(self) -> DataLoader:
        """Return the validation dataloader."""
        assert self.val_dataset is not None, (
            "Val dataset not initialized. Call setup() first."
        )
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=False,
            pin_memory=True,
            prefetch_factor=2 if self.num_workers > 0 else None,
        )

    def test_dataloader(self) -> DataLoader:
        """Return the test dataloader."""
        assert self.test_dataset is not None, (
            "Test dataset not initialized. Call setup() first."
        )
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=min(self.num_workers, 2),
        )

    @property
    def sample_shapes(self) -> dict:
        """Return the shapes of samples for this dataset."""
        return {
            "sequence": (self.sequence_length,),
            "mask": (self.sequence_length,),
            "cls": (),
        }

    @property
    def belief_shapes(self) -> dict:
        """Return the shapes for belief representation."""
        return {
            "sequence": (self.sequence_length, self.alphabet_size),
        }

    def decode_sequence(self, seq_tensor: torch.Tensor) -> str:
        """
        Decode a sequence tensor to a DNA string.

        Args:
            seq_tensor: Integer-encoded sequence tensor

        Returns:
            DNA sequence as string (A, C, G, T)
        """
        nucleotides = ["A", "C", "G", "T"]
        return "".join(nucleotides[i] for i in seq_tensor.tolist())


class ToySequenceDataset(Dataset):
    """
    Toy dataset for benchmarking sequence generation models.
    
    Generates sequences from position-dependent categorical distributions,
    similar to the toy dataset in fisher-flow. This is useful for evaluating
    model performance on a controlled synthetic task.
    """

    def __init__(
        self,
        probs: torch.Tensor,
        seq_len: int,
        alphabet_size: int,
        size: int = 100_000,
    ):
        """
        Initialize the toy sequence dataset.
        
        Args:
            probs: Ground truth probability distributions, shape (seq_len, alphabet_size)
            seq_len: Length of sequences
            alphabet_size: Size of the alphabet (number of categories)
            size: Number of samples in the dataset
        """
        self.probs = probs
        self.seq_len = seq_len
        self.alphabet_size = alphabet_size
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx) -> SequenceSample:
        """
        Generate a single sequence sample.
        
        Returns:
            SequenceSample with sequence, mask, and cls (dummy class label)
        """
        # Sample categorical indices from position-specific distributions
        sample = torch.multinomial(
            replacement=True, num_samples=1, input=self.probs
        ).squeeze(-1)
        # sample shape: (seq_len,) - one index per position
        
        # Create mask (all positions are valid)
        mask: torch.BoolTensor = torch.ones_like(sample, dtype=torch.bool)  # type: ignore
        
        # Dummy class label (toy sequences are unconditional)
        cls = torch.tensor(0, dtype=torch.long)
        
        return {
            "sequence": sample,
            "mask": mask,
            "cls": cls,
        }


class ToySequenceDataModule(pl.LightningDataModule):
    """
    A Lightning DataModule for toy sequence generation benchmarks.
    
    This module creates synthetic sequences from position-dependent categorical
    distributions, matching the toy dataset design from fisher-flow. The ground
    truth distribution is known, allowing precise evaluation of generation quality
    via KL divergence.
    """

    @classmethod
    def from_config(
        cls,
        seq_len: int = 4,
        alphabet_size: int = 100,
        batch_size: int = 512,
        num_workers: int = 4,
        train_val_test_split: tuple[int, int, int] = (100_000, 5_000, 10_000),
        seed: int = 42,
        **kwargs,
    ) -> "ToySequenceDataModule":
        """
        Create a ToySequenceDataModule from configuration parameters.

        Args:
            seq_len: Length of sequences (k in fisher-flow)
            alphabet_size: Size of alphabet (dim in fisher-flow)
            batch_size: Batch size for dataloaders
            num_workers: Number of workers for dataloaders
            train_val_test_split: Sizes of train/val/test splits
            seed: Random seed for reproducibility
            **kwargs: Additional arguments (ignored for compatibility)

        Returns:
            ToySequenceDataModule instance
        """
        return cls(
            seq_len=seq_len,
            alphabet_size=alphabet_size,
            batch_size=batch_size,
            num_workers=num_workers,
            train_val_test_split=train_val_test_split,
            seed=seed,
        )

    def __init__(
        self,
        seq_len: int = 4,
        alphabet_size: int = 100,
        batch_size: int = 512,
        num_workers: int = 4,
        train_val_test_split: tuple[int, int, int] = (100_000, 5_000, 10_000),
        seed: int = 42,
    ):
        """
        Initialize the ToySequenceDataModule.

        Args:
            seq_len: Length of sequences
            alphabet_size: Size of the alphabet
            batch_size: Batch size for dataloaders
            num_workers: Number of workers for dataloaders
            train_val_test_split: Sizes of train/val/test splits
            seed: Random seed for reproducibility
        """
        super().__init__()
        self.seq_len = seq_len
        self.alphabet_size = alphabet_size
        self.batch_size = batch_size
        self.num_workers = min(num_workers, available_cpu_count())
        self.train_val_test_split = train_val_test_split
        self.seed = seed

        # Create ground truth probability distributions (position-dependent)
        torch.manual_seed(seed)
        self.probs = torch.softmax(
            torch.rand(seq_len, alphabet_size), dim=-1
        )

        # Metadata
        self.num_cls = 1  # Toy sequences are unconditional
        self.sequence_length = seq_len

        # Datasets
        self.train_dataset: ToySequenceDataset | None = None
        self.val_dataset: ToySequenceDataset | None = None
        self.test_dataset: ToySequenceDataset | None = None

    def prepare_data(self):
        """Nothing to download for toy data."""
        pass

    def setup(self, stage: str | None = None):
        """
        Create datasets. Called on every GPU/process.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'
        """
        if stage == "fit" or stage is None:
            self.train_dataset = ToySequenceDataset(
                probs=self.probs,
                seq_len=self.seq_len,
                alphabet_size=self.alphabet_size,
                size=self.train_val_test_split[0],
            )
            self.val_dataset = ToySequenceDataset(
                probs=self.probs,
                seq_len=self.seq_len,
                alphabet_size=self.alphabet_size,
                size=self.train_val_test_split[1],
            )

        if stage == "test" or stage is None:
            self.test_dataset = ToySequenceDataset(
                probs=self.probs,
                seq_len=self.seq_len,
                alphabet_size=self.alphabet_size,
                size=self.train_val_test_split[2],
            )

    def train_dataloader(self) -> DataLoader:
        """Return the training dataloader."""
        assert self.train_dataset is not None
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=True,
            prefetch_factor=2 if self.num_workers > 0 else None,
        )

    def val_dataloader(self) -> DataLoader:
        """Return the validation dataloader."""
        assert self.val_dataset is not None
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=False,
            pin_memory=True,
            prefetch_factor=2 if self.num_workers > 0 else None,
        )

    def test_dataloader(self) -> DataLoader:
        """Return the test dataloader."""
        assert self.test_dataset is not None
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=min(self.num_workers, 2),
        )

    @property
    def sample_shapes(self) -> dict:
        """Return the shapes of samples for this dataset."""
        return {
            "sequence": (self.sequence_length,),
            "mask": (self.sequence_length,),
            "cls": (),
        }

    @property
    def belief_shapes(self) -> dict:
        """Return the shapes for belief representation."""
        return {
            "sequence": (self.sequence_length, self.alphabet_size),
        }

    def decode_sequence(self, seq_tensor: torch.Tensor) -> str:
        """
        Decode a sequence tensor to a string representation.

        Args:
            seq_tensor: Integer-encoded sequence tensor

        Returns:
            Sequence as string
        """
        return "".join(str(i.item()) for i in seq_tensor)
