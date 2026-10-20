"""
Frechet Biological Distance (FBD) Benchmarker for DNA Sequence Design

This module provides utilities to compute FBD metrics for evaluating generated
DNA sequences against reference sequences using pretrained classifiers.

Based on the paper: "Dirichlet Flow Matching with Applications to DNA Sequence Design"
(arXiv:2402.05841)
"""

import os
from pathlib import Path
import re
from typing import TYPE_CHECKING, Union, Optional, Tuple, Dict, List
import warnings

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import yaml
from scipy.linalg import sqrtm
import lightning.pytorch as pl
import wandb

from project.utils.metrics_capture import record_many

if TYPE_CHECKING:
    from project.data.sequence import ToySequenceDataModule
from project.reconstructor.sequence import (
    CNNModel,
)  # , MLPModel, TransformerModel, DeepFlyBrainModel

# from utils.dataset import EnhancerDataset
import pickle
from project.data.sequence import SequenceDataModule
from project.samplers.sampler import Sampler
from project.tasks import ELBO
import copy


DEFAULT_DNA_CACHE_DIR = "data/dna"
DEFAULT_DNA_WORKDIR = "data/dna/workdir"

FLYBRAIN_CLS_DIR = "clsDNAclean_cnn_1stack_2023-12-30_15-01-30"
MEL2_CLS_DIR = "clsMELclean_cnn_dropout02_2023-12-31_12-26-28"


def _is_mel_task(task: str) -> bool:
    return task.lower() in {"mel", "mel2", "melanoma"}


def _normalize_task(task: str) -> str:
    if _is_mel_task(task):
        return "MEL2"
    if task.lower() in {"flybrain", "fly"}:
        return "FlyBrain"
    return task


class EnhancerDataset(torch.utils.data.Dataset):
    def __init__(self, mel_enhancer: bool, split: str = "train", cache_dir: str = DEFAULT_DNA_CACHE_DIR):
        dataset_name = "DeepMEL2" if mel_enhancer else "DeepFlyBrain"
        data_path = Path(cache_dir) / f"{dataset_name}_data.pkl"
        if not data_path.exists():
            raise FileNotFoundError(
                f"Enhancer dataset not found: {data_path}\n"
                "Run `python scripts/download_data.py --dna` to download it."
            )
        with open(data_path, "rb") as f:
            all_data = pickle.load(f)
        self.seqs = torch.argmax(
            torch.from_numpy(copy.deepcopy(all_data[f"{split}_data"])), dim=-1
        )
        self.clss = torch.argmax(
            torch.from_numpy(copy.deepcopy(all_data[f"y_{split}"])), dim=-1
        )
        self.num_cls = all_data[f"y_{split}"].shape[-1]
        self.alphabet_size = 4

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return self.seqs[idx], self.clss[idx]


class FBDBenchmarker:
    """Benchmarker for Frechet Biological Distance (FBD) on DNA sequences."""

    TASK_CONFIGS = {
        "FlyBrain": {
            "name": "FlyBrain Enhancer",
            "ckpt_relpath": f"{FLYBRAIN_CLS_DIR}/epoch=15-step=10480.ckpt",
            "hparams_relpath": f"{FLYBRAIN_CLS_DIR}/lightning_logs/version_0/hparams.yaml",
            "num_classes": 81,
            "target_class": 2,
            "sequence_length": 500,
            "dataset_name": "DeepFlyBrain",
        },
        "MEL2": {
            "name": "MEL (Melanoma) Enhancer",
            "ckpt_relpath": f"{MEL2_CLS_DIR}/epoch=9-step=5540.ckpt",
            "hparams_relpath": f"{MEL2_CLS_DIR}/lightning_logs/version_0/hparams.yaml",
            "num_classes": 47,
            "target_class": 13,
            "sequence_length": 500,
            "dataset_name": "DeepMEL2",
        },
    }

    def __init__(
        self,
        task: str = "FlyBrain",
        ckpt_path: Optional[str] = None,
        hparams_path: Optional[str] = None,
        device: Optional[str] = None,
        workdir: str = DEFAULT_DNA_WORKDIR,
        cache_dir: str = DEFAULT_DNA_CACHE_DIR,
    ):
        task = _normalize_task(task)
        if task not in self.TASK_CONFIGS:
            raise ValueError(
                f"Unknown task '{task}'. Must be one of: {list(self.TASK_CONFIGS.keys())}"
            )

        self.task = task
        self.config = self.TASK_CONFIGS[task].copy()
        self.workdir = workdir
        self.cache_dir = cache_dir

        if ckpt_path is not None:
            self.config["ckpt_path"] = ckpt_path
        else:
            self.config["ckpt_path"] = os.path.join(workdir, self.config["ckpt_relpath"])

        if hparams_path is not None:
            self.config["hparams_path"] = hparams_path
        else:
            self.config["hparams_path"] = os.path.join(
                workdir, self.config["hparams_relpath"]
            )

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.classifier = None
        self.hparams = None
        self.num_classes = self.config["num_classes"]
        self.target_class = self.config["target_class"]
        self.sequence_length = self.config["sequence_length"]

        self._load_classifier()

    def _load_classifier(self):
        """Load the pretrained classifier model."""
        ckpt_path = self.config["ckpt_path"]
        hparams_path = self.config["hparams_path"]

        # Check if files exist
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"Checkpoint not found: {ckpt_path}\n"
                f"Please download pretrained models from:\n"
                f"https://publbuck.s3.us-east-2.amazonaws.com/workdir.zip\n"
                f"and extract to '{self.workdir}/' directory"
            )

        if not os.path.exists(hparams_path):
            raise FileNotFoundError(
                f"Hyperparameters file not found: {hparams_path}\n"
                f"Please download pretrained models from:\n"
                f"https://publbuck.s3.us-east-2.amazonaws.com/workdir.zip\n"
                f"and extract to '{self.workdir}/' directory"
            )

        # Load hyperparameters
        with open(hparams_path, "r") as f:
            self.hparams = yaml.load(f, Loader=yaml.UnsafeLoader)

        args = self.hparams["args"]

        # Determine model type from hyperparameters
        model_type = (
            "cnn"  # getattr(args, 'clean_cls_model', getattr(args, 'model', 'cnn'))
        )

        # Create the classifier model
        if model_type == "cnn":
            self.classifier = CNNModel(
                args, alphabet_size=4, num_cls=self.num_classes, classifier=True
            )
        # elif model_type == 'mlp':
        #     self.classifier = MLPModel(
        #         args,
        #         alphabet_size=4,
        #         num_cls=self.num_classes,
        #         classifier=True
        #     )
        # elif model_type == 'transformer':
        #     self.classifier = TransformerModel(
        #         args,
        #         alphabet_size=4,
        #         num_cls=self.num_classes,
        #         classifier=True
        #     )
        # elif model_type == 'deepflybrain':
        #     self.classifier = DeepFlyBrainModel(
        #         args,
        #         alphabet_size=4,
        #         num_cls=self.num_classes,
        #         classifier=True
        #     )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Load checkpoint weights
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint["state_dict"]

        # Remove 'model.' prefix from keys if present
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace("model.", "")
            new_state_dict[new_key] = v

        self.classifier.load_state_dict(new_state_dict)
        self.classifier.eval()
        self.classifier.to(self.device)

        # Freeze parameters
        for param in self.classifier.parameters():
            param.requires_grad = False

        print(f"✓ Loaded {self.config['name']} classifier")
        print(f"  Model: {model_type.upper()}")
        print(f"  Classes: {self.num_classes}")
        print(f"  Target class: {self.target_class}")
        print(f"  Device: {self.device}")

    def load_reference_dataset(
        self, split: str = "test", max_samples: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load reference sequences from the dataset.

        Args:
            split: Dataset split ('train', 'valid', or 'test')
            max_samples: Maximum number of samples to load (None for all)

        Returns:
            sequences: Integer-encoded sequences, shape (N, sequence_length)
            labels: Class labels, shape (N,)
        """

        class DatasetArgs:
            mel_enhancer = _is_mel_task(self.task)

        dataset = EnhancerDataset(
            mel_enhancer=DatasetArgs.mel_enhancer,
            split=split,
            cache_dir=self.cache_dir,
        )

        sequences = dataset.seqs
        labels = dataset.clss

        if max_samples is not None and len(sequences) > max_samples:
            indices = torch.randperm(len(sequences))[:max_samples]
            sequences = sequences[indices]
            labels = labels[indices]

        return sequences, labels

    def extract_embeddings(
        self,
        sequences: Union[np.ndarray, torch.Tensor],
        batch_size: int = 256,
        verbose: bool = True,
    ) -> np.ndarray:
        """
        Extract embeddings from sequences using the classifier.

        Args:
            sequences: DNA sequences, shape (N, sequence_length).
                      Integer-encoded (A=0, C=1, G=2, T=3)
            batch_size: Batch size for processing
            verbose: Print progress information

        Returns:
            Embeddings array, shape (N, embedding_dim)
        """
        # Convert to torch tensor if needed
        if isinstance(sequences, np.ndarray):
            sequences = torch.from_numpy(sequences)

        sequences = sequences.long()

        all_embeddings = []
        num_batches = (len(sequences) + batch_size - 1) // batch_size

        with torch.no_grad():
            for i in range(0, len(sequences), batch_size):
                if verbose and i // batch_size % max(1, num_batches // 10) == 0:
                    progress = (i / len(sequences)) * 100
                    print(f"  Extracting embeddings: {progress:.1f}%", end="\r")

                batch = sequences[i : i + batch_size].to(self.device)
                batch_size_actual = len(batch)

                # Time parameter set to 0 for clean sequences
                t = torch.zeros(batch_size_actual, device=self.device)

                # Extract embeddings from the classifier
                _, embeddings = self.classifier(batch, t=t, return_embedding=True)

                all_embeddings.append(embeddings.cpu().numpy())

        if verbose:
            print("  Extracting embeddings: 100.0%")

        return np.concatenate(all_embeddings, axis=0)

    def compute_fbd(
        self,
        sequences_1: Union[np.ndarray, torch.Tensor],
        sequences_2: Union[np.ndarray, torch.Tensor],
        batch_size: int = 256,
        return_embeddings: bool = False,
        verbose: bool = True,
    ) -> Union[float, Tuple[float, np.ndarray, np.ndarray]]:
        """
        Compute Frechet Biological Distance between two sets of sequences.

        Args:
            sequences_1: First set of sequences, shape (N1, sequence_length)
            sequences_2: Second set of sequences, shape (N2, sequence_length)
            batch_size: Batch size for processing
            return_embeddings: If True, also return the embeddings
            verbose: Print progress information

        Returns:
            fbd: The Frechet Biological Distance (float)
            If return_embeddings=True: (fbd, embeddings_1, embeddings_2)

        Example:
            >>> benchmarker = FBDBenchmarker(task='flybrain')
            >>> generated_seqs = torch.randint(0, 4, (1000, 200))
            >>> reference_seqs, _ = benchmarker.load_reference_dataset('test')
            >>> fbd = benchmarker.compute_fbd(generated_seqs, reference_seqs)
            >>> print(f"FBD: {fbd:.4f}")
        """
        if verbose:
            print(f"\nComputing FBD for {self.config['name']}")
            print(f"  Generated sequences: {len(sequences_1)}")
            print(f"  Reference sequences: {len(sequences_2)}")

        # Extract embeddings
        if verbose:
            print("  Step 1/2: Extracting embeddings for generated sequences")
        embeddings_1 = self.extract_embeddings(sequences_1, batch_size, verbose)

        if verbose:
            print("  Step 2/2: Extracting embeddings for reference sequences")
        embeddings_2 = self.extract_embeddings(sequences_2, batch_size, verbose)

        # Compute FBD
        fbd = self._compute_frechet_distance(embeddings_1, embeddings_2)

        if verbose:
            print(f"\n✓ FBD: {fbd:.4f}")

        if return_embeddings:
            return fbd, embeddings_1, embeddings_2
        return fbd

    def compute_fbd_metrics(
        self,
        generated_sequences: Union[np.ndarray, torch.Tensor],
        reference_split: str = "test",
        target_class_only: bool = True,
        compute_random_baseline: bool = True,
        batch_size: int = 256,
        verbose: bool = True,
    ) -> Dict[str, float]:
        """
        Compute comprehensive FBD metrics as reported in the paper.

        This includes:
        - FBD to all reference sequences
        - FBD to target class sequences only (if target_class_only=True)
        - FBD from random sequences baseline (if compute_random_baseline=True)

        Args:
            generated_sequences: Generated sequences to evaluate
            reference_split: Dataset split to use as reference ('test', 'valid', 'train')
            target_class_only: Also compute FBD to target class only
            compute_random_baseline: Compute baseline with random sequences
            batch_size: Batch size for processing
            verbose: Print progress information

        Returns:
            Dictionary with FBD metrics:
            - 'fbd_to_all': FBD from generated to all reference sequences
            - 'fbd_to_target_class': FBD to target class only (if applicable)
            - 'fbd_random_to_all': Baseline FBD from random sequences (if applicable)

        Example:
            >>> benchmarker = FBDBenchmarker(task='flybrain')
            >>> generated_seqs = torch.randint(0, 4, (1000, 200))
            >>> metrics = benchmarker.compute_fbd_metrics(generated_seqs)
            >>> print(metrics)
        """
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"FBD Evaluation: {self.config['name']}")
            print(f"{'=' * 60}")

        # Load reference dataset
        if verbose:
            print(f"\nLoading reference dataset (split={reference_split})...")
        ref_sequences, ref_labels = self.load_reference_dataset(reference_split)

        if verbose:
            print(f"  ✓ Loaded {len(ref_sequences)} reference sequences")

        metrics = {}

        # Extract embeddings for generated sequences (reuse for all metrics)
        if verbose:
            print(
                f"\nExtracting embeddings for {len(generated_sequences)} generated sequences..."
            )
        gen_embeddings = self.extract_embeddings(generated_sequences, batch_size, verbose)

        # 1. FBD to all reference sequences
        if verbose:
            print("\nComputing FBD to all reference sequences...")
        ref_embeddings = self.extract_embeddings(ref_sequences, batch_size, verbose)
        metrics["fbd_to_all"] = self._compute_frechet_distance(
            gen_embeddings, ref_embeddings
        )

        if verbose:
            print(f"  ✓ FBD to all: {metrics['fbd_to_all']:.4f}")

        # 2. FBD to target class only
        if target_class_only:
            if verbose:
                print(f"\nComputing FBD to target class {self.target_class} only...")

            target_mask = ref_labels == self.target_class
            target_sequences = ref_sequences[target_mask]

            if len(target_sequences) > 0:
                target_embeddings = self.extract_embeddings(
                    target_sequences, batch_size, verbose
                )
                metrics["fbd_to_target_class"] = self._compute_frechet_distance(
                    gen_embeddings, target_embeddings
                )

                if verbose:
                    print(
                        f"  ✓ FBD to target class: {metrics['fbd_to_target_class']:.4f}"
                    )
                    print(
                        f"    (using {len(target_sequences)} sequences from class {self.target_class})"
                    )
            else:
                warnings.warn(f"No sequences found for target class {self.target_class}")
                metrics["fbd_to_target_class"] = float("nan")

        # 3. Random baseline
        if compute_random_baseline:
            if verbose:
                print("\nComputing random sequence baseline...")

            random_sequences = torch.randint(
                0, 4, (len(generated_sequences), self.sequence_length), dtype=torch.long
            )
            random_embeddings = self.extract_embeddings(
                random_sequences, batch_size, verbose
            )
            metrics["fbd_random_to_all"] = self._compute_frechet_distance(
                random_embeddings, ref_embeddings
            )

            if verbose:
                print(f"  ✓ FBD (random baseline): {metrics['fbd_random_to_all']:.4f}")

        # Print summary
        if verbose:
            print(f"\n{'=' * 60}")
            print("Summary:")
            print(f"{'=' * 60}")
            for key, value in metrics.items():
                print(f"  {key:30s}: {value:8.4f}")
            print(f"{'=' * 60}\n")

        return metrics

    @staticmethod
    def _compute_frechet_distance(
        embeddings_1: np.ndarray, embeddings_2: np.ndarray
    ) -> float:
        """
        Compute the Frechet distance between two sets of embeddings.

        The Frechet distance (2-Wasserstein distance) between two
        multivariate Gaussians:

        d² = ||μ₁ - μ₂||² + Tr(Σ₁ + Σ₂ - 2√(Σ₁Σ₂))

        Args:
            embeddings_1: First set of embeddings, shape (N1, embedding_dim)
            embeddings_2: Second set of embeddings, shape (N2, embedding_dim)

        Returns:
            The Frechet distance (float). Returns NaN if inputs contain NaN or are empty.
        """
        # Check for invalid inputs
        if (
            np.isnan(embeddings_1).any()
            or np.isnan(embeddings_2).any()
            or len(embeddings_1) == 0
            or len(embeddings_2) == 0
        ):
            return float("nan")

        # Compute mean and covariance for both distributions
        mu1, sigma1 = embeddings_1.mean(axis=0), np.cov(embeddings_1, rowvar=False)
        mu2, sigma2 = embeddings_2.mean(axis=0), np.cov(embeddings_2, rowvar=False)

        # Compute squared difference between means
        mean_diff_squared = np.sum((mu1 - mu2) ** 2.0)

        # Compute matrix square root of the product of covariances
        covmean = sqrtm(sigma1.dot(sigma2))

        # Handle numerical errors (imaginary components from sqrtm)
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        # Compute the Frechet distance
        frechet_dist = mean_diff_squared + np.trace(sigma1 + sigma2 - 2.0 * covmean)

        return float(frechet_dist)

    @staticmethod
    def sequences_to_integer_encoding(
        sequences: List[str], alphabet: Optional[Dict[str, int]] = None
    ) -> np.ndarray:
        """
        Convert DNA sequences from string format to integer encoding.

        Args:
            sequences: List of DNA sequence strings (e.g., ['ATCG', 'GCTA', ...])
            alphabet: Mapping from nucleotide to integer.
                     Default: {'A': 0, 'C': 1, 'G': 2, 'T': 3}

        Returns:
            Integer-encoded sequences as numpy array, shape (N, sequence_length)

        Example:
            >>> seqs = ['ATCG', 'GCTA']
            >>> encoded = FBDBenchmarker.sequences_to_integer_encoding(seqs)
            >>> print(encoded)
            [[0 3 1 2]
             [2 1 3 0]]
        """
        if alphabet is None:
            alphabet = {"A": 0, "C": 1, "G": 2, "T": 3}

        encoded = []
        for seq in sequences:
            encoded.append([alphabet[nuc.upper()] for nuc in seq])

        return np.array(encoded, dtype=np.int64)

    @staticmethod
    def integer_to_sequence_encoding(
        sequences: Union[np.ndarray, torch.Tensor],
        alphabet: Optional[Dict[int, str]] = None,
    ) -> List[str]:
        """
        Convert integer-encoded DNA sequences to string format.

        Args:
            sequences: Integer-encoded sequences, shape (N, sequence_length)
            alphabet: Mapping from integer to nucleotide.
                     Default: {0: 'A', 1: 'C', 2: 'G', 3: 'T'}

        Returns:
            List of DNA sequence strings

        Example:
            >>> encoded = np.array([[0, 3, 1, 2], [2, 1, 3, 0]])
            >>> seqs = FBDBenchmarker.integer_to_sequence_encoding(encoded)
            >>> print(seqs)
            ['ATCG', 'GCTA']
        """
        if alphabet is None:
            alphabet = {0: "A", 1: "C", 2: "G", 3: "T"}

        if isinstance(sequences, torch.Tensor):
            sequences = sequences.cpu().numpy()

        string_sequences = []
        for seq in sequences:
            string_sequences.append("".join([alphabet[int(nuc)] for nuc in seq]))

        return string_sequences

    def get_config(self) -> Dict:
        """Get the current configuration."""
        return self.config.copy()

    def __repr__(self) -> str:
        return (
            f"FBDBenchmarker(\n"
            f"  task='{self.task}',\n"
            f"  name='{self.config['name']}',\n"
            f"  num_classes={self.num_classes},\n"
            f"  target_class={self.target_class},\n"
            f"  device='{self.device}'\n"
            f")"
        )


def compute_frechet_biological_distance(
    sequences_1: Union[np.ndarray, torch.Tensor],
    sequences_2: Union[np.ndarray, torch.Tensor],
    classifier_model: nn.Module,
    device: str = "cuda",
    batch_size: int = 256,
    return_embeddings: bool = False,
    verbose: bool = False,
) -> Union[float, Tuple[float, np.ndarray, np.ndarray]]:
    """
    Standalone function to compute FBD with a custom classifier.

    This is a lower-level function if you already have a loaded classifier.
    For most use cases, prefer using the FBDBenchmarker class.

    Args:
        sequences_1: First set of DNA sequences, shape (N1, sequence_length)
        sequences_2: Second set of DNA sequences, shape (N2, sequence_length)
        classifier_model: Pretrained classifier with return_embedding support
        device: Device to run on
        batch_size: Batch size for processing
        return_embeddings: Whether to return embeddings
        verbose: Print progress

    Returns:
        fbd: The Frechet Biological Distance
        If return_embeddings=True: (fbd, embeddings_1, embeddings_2)
    """
    # Convert to torch tensors if needed
    if isinstance(sequences_1, np.ndarray):
        sequences_1 = torch.from_numpy(sequences_1)
    if isinstance(sequences_2, np.ndarray):
        sequences_2 = torch.from_numpy(sequences_2)

    sequences_1 = sequences_1.long()
    sequences_2 = sequences_2.long()

    classifier_model = classifier_model.to(device)
    classifier_model.eval()

    def extract_embeddings(sequences: torch.Tensor) -> np.ndarray:
        all_embeddings = []
        with torch.no_grad():
            for i in range(0, len(sequences), batch_size):
                batch = sequences[i : i + batch_size].to(device)
                t = torch.zeros(len(batch), device=device)
                _, embeddings = classifier_model(batch, t=t, return_embedding=True)
                all_embeddings.append(embeddings.cpu().numpy())
        return np.concatenate(all_embeddings, axis=0)

    if verbose:
        print("Extracting embeddings...")
    embeddings_1 = extract_embeddings(sequences_1)
    embeddings_2 = extract_embeddings(sequences_2)

    fbd = FBDBenchmarker._compute_frechet_distance(embeddings_1, embeddings_2)

    if return_embeddings:
        return fbd, embeddings_1, embeddings_2
    return fbd


class SequenceScorePlotter(pl.Callback):
    """
    PyTorch Lightning callback for computing and logging FBD metrics during training.

    This callback integrates the FBDBenchmarker into the training loop, computing
    FBD metrics on generated sequences at the end of each validation epoch.
    """

    @classmethod
    def from_config(
        cls,
        datamodule: "SequenceDataModule",
        task: str = "FlyBrain",
        num_samples: int = 1000,
        batch_size: int = 256,
        compute_target_class_fbd: bool = True,
        compute_random_baseline: bool = True,
        reference_split: str = "test",
        plot_every_n_epochs: int = 10,
        workdir: str = DEFAULT_DNA_WORKDIR,
        cache_dir: str = DEFAULT_DNA_CACHE_DIR,
        **kwargs,
    ):
        return cls(
            datamodule=datamodule,
            task=task,
            num_samples=num_samples,
            batch_size=batch_size,
            compute_target_class_fbd=compute_target_class_fbd,
            compute_random_baseline=compute_random_baseline,
            reference_split=reference_split,
            workdir=workdir,
            cache_dir=cache_dir,
        )

    def __init__(
        self,
        datamodule: "SequenceDataModule",
        task: str = "flybrain",
        num_samples: int = 1000,
        batch_size: int = 256,
        compute_target_class_fbd: bool = True,
        compute_random_baseline: bool = True,
        reference_split: str = "test",
        workdir: str = DEFAULT_DNA_WORKDIR,
        cache_dir: str = DEFAULT_DNA_CACHE_DIR,
    ):
        super().__init__()
        self.datamodule = datamodule
        self.task = _normalize_task(task)
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.compute_target_class_fbd = compute_target_class_fbd
        self.compute_random_baseline = compute_random_baseline
        self.reference_split = reference_split
        self.workdir = workdir
        self.cache_dir = cache_dir

        # Initialize benchmarker (will be created when needed to ensure correct device)
        self.benchmarker: Optional[FBDBenchmarker] = None

    def _ensure_benchmarker(self, device: str):
        """Lazy initialization of benchmarker with correct device."""
        if self.benchmarker is None:
            print(f"Initializing FBD benchmarker for task: {self.task}")
            self.benchmarker = FBDBenchmarker(
                task=self.task,
                device=device,
                workdir=self.workdir,
                cache_dir=self.cache_dir,
            )

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Called at the end of validation epoch."""

        # Check if this is an ELBO task
        if not isinstance(pl_module, ELBO):
            print("SequenceScorePlotter is only compatible with ELBO tasks")
            return

        # Skip epoch 0 and check epoch interval
        current_epoch = trainer.current_epoch
        # if current_epoch == 0:
        #     print("Skipping FBD evaluation at epoch 0")
        #     return

        # if current_epoch % self.plot_every_n_epochs != 0:
        #     return

        print(f"\n{'=' * 60}")
        print(f"Computing FBD metrics at epoch {current_epoch}")
        print(f"{'=' * 60}")

        # Initialize benchmarker with correct device
        device = str(pl_module.device)
        self._ensure_benchmarker(device)

        # Generate sequences for each sampler
        for sampler in pl_module.samplers:
            sampler_name = sampler.name
            print(f"\nGenerating sequences for sampler: {sampler_name}")

            try:
                # Generate sequences
                generated_sequences = self._generate_sequences(
                    sampler, pl_module, self.num_samples
                )

                if generated_sequences is None or len(generated_sequences) == 0:
                    print(f"No valid sequences generated by {sampler_name}")
                    continue

                print(f"Generated {len(generated_sequences)} sequences")
                artifact_name = re.sub(
                    r"[^a-zA-Z0-9._-]",
                    "_",
                    f"generated_sequences_{sampler_name}_epoch_{pl_module.trainer.current_epoch}",
                )
                print(f"Logging generated SMILES to artifact: {artifact_name}")
                smiles_artifact = wandb.Artifact(
                    artifact_name,
                    type="generated_sequences",
                    description=f"Generated SMILES from {sampler_name} sampler",
                )
                sequence_strings = [
                    self.datamodule.decode_sequence(seq) for seq in generated_sequences
                ]
                print(f"First 2 generated sequences: {sequence_strings[:2]}")
                smiles_table = wandb.Table(
                    data=[[seq] for seq in sequence_strings],
                    columns=["sequences"],
                )
                smiles_artifact.add(
                    smiles_table,
                    "generated_sequences",
                )
                wandb.log({f"tables/generated_sequences/{sampler_name}": smiles_table})
                wandb.log_artifact(smiles_artifact)

                if self.benchmarker is None:
                    print("FBD benchmarker not initialized")
                    continue
                # Compute FBD metrics
                metrics = self.benchmarker.compute_fbd_metrics(
                    generated_sequences=generated_sequences,
                    reference_split=self.reference_split,
                    target_class_only=self.compute_target_class_fbd,
                    compute_random_baseline=self.compute_random_baseline,
                    batch_size=self.batch_size,
                    verbose=True,
                )

                # Log metrics to wandb
                for metric_name, metric_value in metrics.items():
                    log_key = f"val/{sampler_name}/{metric_name}"
                    print(f"Logging {log_key}: {metric_value:.4f}")
                    record_many({log_key: metric_value})
                    wandb.log({log_key: metric_value})

            except Exception as e:
                print(f"Error computing FBD metrics for {sampler_name}: {e}")
                import traceback

                traceback.print_exc()

        print(f"{'=' * 60}\n")

    def _generate_sequences(
        self,
        sampler: Sampler,
        pl_module: pl.LightningModule,
        num_samples: int,
    ) -> Optional[torch.Tensor]:
        """
        Generate sequences using the sampler.

        Returns:
            Tensor of integer-encoded sequences, shape (num_samples, sequence_length)
        """
        all_sequences = []
        num_batches = (num_samples + self.batch_size - 1) // self.batch_size

        print(f"Generating {num_samples} sequences in {num_batches} batches...")

        for batch_idx in tqdm(range(num_batches), desc="Generating sequences"):
            current_batch_size = min(self.batch_size, num_samples - len(all_sequences))

            # Sample from the model
            batch, _belief_series = sampler.sample_batch(batch_size=current_batch_size)

            # Extract the sequence information
            # Assuming the sequence is in samples['sequence'] and is one-hot or categorical
            if "sequence" in batch:
                seq = batch["sequence"]
                # If one-hot, convert to integer encoding
                if seq.dim() == 3:  # (batch, length, num_classes)
                    seq = seq.argmax(dim=-1)
                all_sequences.append(seq.cpu())
            else:
                print(
                    f"Warning: 'sequence' key not found in batch. Available keys: {batch.keys()}"
                )
                return None

        if len(all_sequences) == 0:
            return None

        sequences = torch.cat(all_sequences, dim=0)[:num_samples]
        return sequences


class ToySequenceScorePlotter(pl.Callback):
    """
    PyTorch Lightning callback for computing and logging KL divergence for toy sequences.

    This callback computes the KL divergence between the empirical distribution of
    generated sequences and the known ground truth distribution, following the approach
    from fisher-flow.
    """

    @classmethod
    def from_config(
        cls,
        datamodule: "ToySequenceDataModule",
        num_samples: int = 10000,
        batch_size: int = 512,
        plot_every_n_epochs: int = 10,
        **kwargs,
    ):
        """
        Create ToySequenceScorePlotter from config.

        Args:
            datamodule: ToySequenceDataModule for the toy dataset
            num_samples: Number of sequences to generate for KL estimation
            batch_size: Batch size for generation
            plot_every_n_epochs: Frequency of evaluation (epochs)
            **kwargs: Additional arguments (ignored for compatibility)
        """
        return cls(
            datamodule=datamodule,
            num_samples=num_samples,
            batch_size=batch_size,
            plot_every_n_epochs=plot_every_n_epochs,
        )

    def __init__(
        self,
        datamodule: "ToySequenceDataModule",
        num_samples: int = 10000,
        batch_size: int = 512,
        plot_every_n_epochs: int = 10,
    ):
        """
        Initialize the ToySequenceScorePlotter.

        Args:
            datamodule: ToySequenceDataModule containing ground truth distributions
            num_samples: Number of sequences to generate for KL estimation
            batch_size: Batch size for generation
            plot_every_n_epochs: Frequency of evaluation
        """
        super().__init__()
        self.datamodule = datamodule
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.plot_every_n_epochs = plot_every_n_epochs

        # Ground truth distribution (position-dependent categorical)
        self.true_probs = datamodule.probs  # Shape: (seq_len, alphabet_size)

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Called at the end of validation epoch."""

        # Check if this is an ELBO task
        if not isinstance(pl_module, ELBO):
            print("ToySequenceScorePlotter is only compatible with ELBO tasks")
            return

        current_epoch = trainer.current_epoch


        print(f"\n{'=' * 60}")
        print(f"Computing KL divergence at epoch {current_epoch}")
        print(f"{'=' * 60}")

        # Generate sequences for each sampler
        for sampler in pl_module.samplers:
            sampler_name = sampler.name
            print(f"\nEvaluating sampler: {sampler_name}")

            try:
                # Generate sequences
                generated_sequences = self._generate_sequences(
                    sampler, pl_module, self.num_samples
                )

                if generated_sequences is None or len(generated_sequences) == 0:
                    print(f"No valid sequences generated by {sampler_name}")
                    continue

                print(f"Generated {len(generated_sequences)} sequences")

                # Compute KL divergence
                kl_forward, kl_reverse = self._compute_kl_divergence(
                    generated_sequences
                )

                # Log metrics
                print(f"  KL(generated || true): {kl_forward:.6f}")
                print(f"  KL(true || generated): {kl_reverse:.6f}")

                # Log to wandb
                metrics = {
                    f"val/{sampler_name}/kl_forward": kl_forward,
                    f"val/{sampler_name}/kl_reverse": kl_reverse,
                }
                record_many(metrics)
                wandb.log(metrics)

                # Log sample sequences
                if wandb.run is not None:
                    artifact_name = re.sub(
                        r"[^a-zA-Z0-9._-]",
                        "_",
                        f"toy_sequences_{sampler_name}_epoch_{current_epoch}",
                    )
                    artifact = wandb.Artifact(
                        artifact_name,
                        type="toy_sequences",
                        description=f"Generated toy sequences from {sampler_name}",
                    )
                    
                    # Sample a few sequences for visualization
                    sample_seqs = generated_sequences[:100]
                    seq_strings = [
                        ",".join(str(i.item()) for i in seq) for seq in sample_seqs
                    ]
                    
                    table = wandb.Table(
                        data=[[seq] for seq in seq_strings],
                        columns=["sequence"],
                    )
                    artifact.add(table, "toy_sequences")
                    wandb.log({f"tables/toy_sequences/{sampler_name}": table})
                    wandb.log_artifact(artifact)

            except Exception as e:
                print(f"Error computing KL for {sampler_name}: {e}")
                import traceback
                traceback.print_exc()

        print(f"{'=' * 60}\n")

    def _generate_sequences(
        self,
        sampler: Sampler,
        pl_module: pl.LightningModule,
        num_samples: int,
    ) -> Optional[torch.Tensor]:
        """
        Generate sequences using the sampler.

        Returns:
            Tensor of integer-encoded sequences, shape (num_samples, seq_len)
        """
        all_sequences = []
        num_batches = (num_samples + self.batch_size - 1) // self.batch_size

        for batch_idx in tqdm(range(num_batches), desc="Generating sequences"):
            current_batch_size = min(self.batch_size, num_samples - len(all_sequences))

            # Sample from the model
            batch, _belief_series = sampler.sample_batch(batch_size=current_batch_size)

            # Extract the sequence
            if "sequence" in batch:
                seq = batch["sequence"]
                # If one-hot/belief representation, convert to integer encoding
                if seq.dim() == 3:  # (batch, length, alphabet_size)
                    seq = seq.argmax(dim=-1)
                all_sequences.append(seq.cpu())
            else:
                print("Warning: 'sequence' key not found in batch")
                return None

        if len(all_sequences) == 0:
            return None

        sequences = torch.cat(all_sequences, dim=0)[:num_samples]
        return sequences

    def _compute_kl_divergence(
        self, sequences: torch.Tensor
    ) -> Tuple[float, float]:
        """
        Compute KL divergence between generated and true distributions.

        Args:
            sequences: Generated integer-encoded sequences, shape (N, seq_len)

        Returns:
            (kl_forward, kl_reverse) where:
                - kl_forward: KL(empirical || true)
                - kl_reverse: KL(true || empirical)
        """
        # Convert to one-hot encoding
        alphabet_size = self.true_probs.shape[1]
        
        one_hot = torch.nn.functional.one_hot(
            sequences.long(), num_classes=alphabet_size
        ).float()
        # Shape: (N, seq_len, alphabet_size)

        # Compute empirical distribution (position-wise counts)
        counts = one_hot.sum(dim=0)  # Shape: (seq_len, alphabet_size)
        empirical_dist = counts / counts.sum(dim=-1, keepdim=True)
        # Shape: (seq_len, alphabet_size)

        # Add small epsilon to avoid log(0)
        eps = 1e-10
        empirical_dist = torch.clamp(empirical_dist, min=eps)
        true_probs = torch.clamp(self.true_probs, min=eps)

        # Compute KL divergences (position-wise, then average)
        # KL(empirical || true) = sum_i empirical(i) * log(empirical(i) / true(i))
        kl_forward = (
            empirical_dist * (torch.log(empirical_dist) - torch.log(true_probs))
        ).sum(dim=-1).mean().item()

        # KL(true || empirical) = sum_i true(i) * log(true(i) / empirical(i))
        kl_reverse = (
            true_probs * (torch.log(true_probs) - torch.log(empirical_dist))
        ).sum(dim=-1).mean().item()

        return kl_forward, kl_reverse
