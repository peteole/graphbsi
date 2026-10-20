import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb

from project.tasks import ELBO


class KLPlotsCallback(pl.Callback):
    @classmethod
    def from_config(cls, datamodule: pl.LightningDataModule, **kwargs):
        return cls(datamodule)

    def __init__(self, datamodule: pl.LightningDataModule, **kwargs):
        super().__init__()
        self.datamodule = datamodule

    def on_validation_epoch_end(self, trainer, pl_module):
        # Compute average KL_by_bucket across all validation batches
        if not isinstance(pl_module, ELBO):
            return
        if pl_module.kl_by_bucket_sums and pl_module.kl_by_bucket_counts:
            avg_kl_by_bucket = (
                (
                    torch.stack(pl_module.kl_by_bucket_sums).sum(dim=0)
                    / torch.stack(pl_module.kl_by_bucket_counts).sum(dim=0)
                )
                .detach()
                .cpu()
                .numpy()
            )
            t = torch.linspace(0, 1, avg_kl_by_bucket.shape[0]).numpy()
            fig, ax = plt.subplots()
            ax.plot(t, avg_kl_by_bucket)
            ax.set_xlabel("Timestep")
            ax.set_ylabel("KL")
            ax.set_title("Average KL by Timestep")
            wandb.log({"val_loss_analysis/KL_by_timestep": wandb.Image(fig)})
            plt.close(fig)
        else:
            print("No KL_by_bucket values to log")
        if len(pl_module.kl_by_component_values) > 0:
            if not isinstance(pl_module.kl_by_component_values[0], dict):
                print("No KL_by_component values to log")
                return
            avg_kl_by_component = {
                k: np.mean([v[k] for v in pl_module.kl_by_component_values])
                for k in pl_module.kl_by_component_values[0].keys()
            }
            fig, ax = plt.subplots()
            ax.bar(avg_kl_by_component.keys(), avg_kl_by_component.values())
            ax.set_xlabel("Component")
            ax.set_ylabel("KL")
            ax.set_title("Average KL by Component")
            ax.set_ylim(bottom=0)
            ax.grid(True, axis="y", linestyle="--", alpha=0.7)
            plt.xticks(rotation=45)
            fig.tight_layout()
            wandb.log({"val_loss_analysis/KL_by_component": wandb.Image(fig)})
            plt.close(fig)
