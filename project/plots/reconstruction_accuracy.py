import math

import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from torch import softmax

from project.belief.belief import Belief
from project.belief.categorical_belief import CategoricalBelief
from project.belief.composed_belief import ComposedBelief
from project.tasks import ELBO
from project.utils import tensor_utils


def get_categorical_beliefs(belief: Belief) -> list[CategoricalBelief]:
    """
    Extracts categorical beliefs from a belief object.

    Args:
        belief (Belief): The belief object to extract categorical beliefs from.

    Returns:
        list[CategoricalBelief]: A list of categorical beliefs.
    """
    categorical_beliefs = []
    if isinstance(belief, CategoricalBelief):
        categorical_beliefs.append(belief)
    elif isinstance(belief, ComposedBelief):
        for b in belief.beliefs:
            categorical_beliefs.extend(get_categorical_beliefs(b))
    return categorical_beliefs


class ReconstructionAccuracyPlot(pl.Callback):
    """
    Callback to plot reconstruction accuracy during training.
    """

    def __init__(
        self,
        datamodule: pl.LightningDataModule,
        num_t_buckets: int = 10,
        min_samples_per_bucket: int = 20,
        **kwargs,
    ):
        super().__init__()
        self.datamodule = datamodule
        self.num_t_buckets = num_t_buckets
        self.min_samples_per_bucket = min_samples_per_bucket

    def on_validation_epoch_end(self, trainer, pl_module):
        if not isinstance(pl_module, ELBO):
            return
        belief = pl_module.belief
        categorical_beliefs = get_categorical_beliefs(belief)
        if len(categorical_beliefs) == 0:
            return
        num_val_samples = len(self.datamodule.val_dataloader().dataset)
        repeat_count = math.ceil(
            self.min_samples_per_bucket * self.num_t_buckets / num_val_samples
        )
        belief_accuracies_by_t_bucket: dict[str, list[list[float]]] = {
            b.key: [[] for _ in range(self.num_t_buckets)] for b in categorical_beliefs
        }
        reconstruction_accuracies_by_t_bucket = {
            b.key: [[] for _ in range(self.num_t_buckets)] for b in categorical_beliefs
        }
        sample_log_probs_belief_by_t_bucket = [[] for _ in range(self.num_t_buckets)]
        sample_log_probs_reconstruction_by_t_bucket = [
            [] for _ in range(self.num_t_buckets)
        ]
        bucket_counts = [0] * self.num_t_buckets
        for batch in self.datamodule.val_dataloader():
            batch = pl_module.transfer_batch_to_device(batch, pl_module.device, 0)
            batch = tensor_utils.dict_of_tensors_to_tensor_dict(batch)
            for _ in range(repeat_count):
                t = torch.rand(batch.batch_size, device=pl_module.device)
                t_buckets = (t * self.num_t_buckets).long()
                b = belief.sample_belief_at_t(batch, t=t)
                b["t"] = t
                if pl_module.class_conditioner is not None:
                    classification = pl_module.class_conditioner.classify_sample(batch)
                    pl_module.class_conditioner.write_sample_classifications_to_belief(
                        classification, b
                    )

                reconstructed = pl_module.reconstructor(b)
                sample_log_probs_belief = torch.zeros(
                    batch.batch_size, device=pl_module.device
                )
                sample_log_probs_reconstruction = torch.zeros(
                    batch.batch_size, device=pl_module.device
                )
                for categorical_belief in categorical_beliefs:
                    belief_probs = softmax(b[categorical_belief.key], dim=-1)
                    reconstructed_probs = reconstructed[categorical_belief.key]
                    actual = batch[categorical_belief.key].long()
                    belief_categories = belief_probs.argmax(dim=-1)
                    reconstructed_categories = reconstructed_probs.argmax(dim=-1)
                    if actual.ndim == belief_categories.ndim + 1:
                        actual = actual.argmax(dim=-1)
                    belief_accuracy = (
                        (belief_categories.long() == actual)
                        .float()
                        .mean(dim=tuple(range(1, actual.ndim)))
                    )
                    reconstructed_accuracy = (
                        (reconstructed_categories.long() == actual)
                        .float()
                        .mean(dim=tuple(range(1, actual.ndim)))
                    )
                    sample_log_probs_belief += torch.gather(
                        belief_probs.log(), dim=-1, index=actual.unsqueeze(-1)
                    ).sum(dim=tuple(range(1, actual.ndim+1)))
                    sample_log_probs_reconstruction += torch.gather(
                        reconstructed_probs.log(), dim=-1, index=actual.unsqueeze(-1)
                    ).sum(dim=tuple(range(1, actual.ndim+1)))
                    for i in range(self.num_t_buckets):
                        mask = t_buckets == i
                        belief_accuracies_by_t_bucket[categorical_belief.key][i].append(
                            belief_accuracy[mask].sum().item()
                        )
                        reconstruction_accuracies_by_t_bucket[categorical_belief.key][
                            i
                        ].append(reconstructed_accuracy[mask].sum().item())

                for i in range(self.num_t_buckets):
                    mask = t_buckets == i
                    bucket_counts[i] += int(mask.sum().item())
                    sample_log_probs_belief_by_t_bucket[i].append(
                        sample_log_probs_belief[mask].sum().item()
                    )
                    sample_log_probs_reconstruction_by_t_bucket[i].append(
                        sample_log_probs_reconstruction[mask].sum().item()
                    )

        for categorical_belief in categorical_beliefs:
            key = categorical_belief.key
            belief_accuracies = belief_accuracies_by_t_bucket[key]
            reconstruction_accuracies = reconstruction_accuracies_by_t_bucket[key]
            t_buckets = np.array(range(self.num_t_buckets)) / self.num_t_buckets
            fig, ax = plt.subplots()
            ax.set_title(f"Misclassification Rate for {key}")
            ax.set_xlabel("t bucket")
            ax.set_ylabel("Miscclassification Rate")
            mean_reconstruction_accuracy_by_t_bucket = [
                np.sum(reconstruction_accuracies[i]) / bucket_counts[i]
                if bucket_counts[i] > 0
                else 0
                for i in range(self.num_t_buckets)
            ]
            mean_belief_accuracy_by_t_bucket = [
                np.sum(belief_accuracies[i]) / bucket_counts[i]
                if bucket_counts[i] > 0
                else 0
                for i in range(self.num_t_buckets)
            ]
            plt.plot(
                t_buckets,
                1 - np.array(mean_reconstruction_accuracy_by_t_bucket),
                label="Reconstruction Misclassification Rate",
            )
            plt.plot(
                t_buckets,
                1 - np.array(mean_belief_accuracy_by_t_bucket),
                label="Belief Missclassification Rate",
            )
            plt.yscale("log")
            plt.legend()
            wandb.log(
                {
                    f"reconstruction_accuracy/misclassification_{key}": wandb.Image(fig),
                }
            )
            plt.close(fig)
        fig, ax = plt.subplots()
        ax.set_title("Sample Log Probabilities")
        ax.set_xlabel("t bucket")
        ax.set_ylabel("Log Probability")
        t_buckets = np.array(range(self.num_t_buckets)) / self.num_t_buckets
        mean_sample_log_probs_belief_by_t_bucket = [
            np.sum(sample_log_probs_belief_by_t_bucket[i])
            / bucket_counts[i]
            if bucket_counts[i] > 0
            else 0
            for i in range(self.num_t_buckets)
        ]
        mean_sample_log_probs_reconstruction_by_t_bucket = [
            np.sum(sample_log_probs_reconstruction_by_t_bucket[i])
            / bucket_counts[i]
            if bucket_counts[i] > 0
            else 0
            for i in range(self.num_t_buckets)
        ]
        plt.plot(
            t_buckets,
            mean_sample_log_probs_belief_by_t_bucket,
            label="Belief",
        )
        plt.plot(
            t_buckets,
            mean_sample_log_probs_reconstruction_by_t_bucket,
            label="Reconstruction",
        )
        plt.legend()
        wandb.log(
            {
                "reconstruction_accuracy/sample_log_probabilities": wandb.Image(fig),
            }
        )
        plt.close(fig)
