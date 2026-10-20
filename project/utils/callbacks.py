from copy import deepcopy
from typing import Any

import lightning.pytorch as pl
import torch
import wandb
from lightning import Callback
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf


class AlwaysSaveCheckpoints(Callback):
    """Log model checkpoints even if training failed.

    As of 04/09/2024, WandbLogger only saves checkpoints on successful runs.
    """

    def on_exception(self, trainer, pl_module, exception):
        for logger in trainer.loggers:
            if isinstance(logger, WandbLogger):
                logger._scan_and_log_checkpoints(logger._checkpoint_callbacks)


class ConfigInCheckpoint(Callback):
    """Save the config in the checkpoint."""

    def __init__(self, config: DictConfig):
        super().__init__()

        self.config = config

    def on_save_checkpoint(self, trainer, pl_module, checkpoint: dict[str, Any]):
        checkpoint["config"] = OmegaConf.to_container(self.config, resolve=True)


class WandbSummaries(pl.Callback):
    """Set the W&B summaries of each metric to the values from the best epoch."""

    def __init__(self, monitor: str, mode: str):
        super().__init__()

        self.monitor = monitor
        self.mode = mode

        self.best_metric = None
        self.best_metrics = None

        self.ready = True

    def on_sanity_check_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self.ready = False

    def on_sanity_check_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self.ready = True

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if not self.ready:
            return

        metrics = trainer.logged_metrics
        if self.monitor in metrics:
            metric = metrics[self.monitor]
            if torch.is_tensor(metric):
                metric = metric.item()

            if self._better(metric):
                self.best_metric = metric
                self.best_metrics = deepcopy(metrics)

        self._update_summaries()

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self._update_summaries()

    def state_dict(self):
        return {
            "monitor": self.monitor,
            "mode": self.mode,
            "best_metric": self.best_metric,
            "best_metrics": self.best_metrics,
        }

    def load_state_dict(self, state_dict):
        self.monitor = state_dict["monitor"]
        self.mode = state_dict["mode"]
        self.best_metric = state_dict["best_metric"]
        self.best_metrics = state_dict["best_metrics"]

    def _better(self, metric):
        if self.best_metric is None:
            return True
        elif self.mode == "min" and metric < self.best_metric:
            return True
        elif self.mode == "max" and metric > self.best_metric:
            return True
        else:
            return False

    def _update_summaries(self):
        # wandb is supposed not to update the summaries anymore once we set them manually,
        # but they are still getting updated, so we make sure to set them after logging
        if self.best_metrics is not None:
            wandb.summary.update(self.best_metrics)
