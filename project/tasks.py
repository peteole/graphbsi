import math
from typing import Optional

import lightning.pytorch as pl
import torch
import torchmetrics as tm
from hydra.utils import instantiate
from omegaconf import DictConfig
from tensordict import TensorDict  # type: ignore

from project.belief.belief import Belief
from project.belief.composed_belief import ComposedBelief
from project.class_conditioner.class_conditioner import ClassConditioner
from project.conditioner.conditioner import Conditioner
from project.reconstructor.reconstructor import SampleReconstructor
from project.samplers.sampler import Sampler
from project.utils.tensor_utils import dict_of_tensors_to_tensor_dict


class ELBO(pl.LightningModule):
    @classmethod
    def from_config(
        cls,
        reconstructor: SampleReconstructor,
        belief: Belief,
        sampler: DictConfig,
        optimizer: DictConfig,
        scheduler: DictConfig,
        datamodule: pl.LightningDataModule,
        ema: DictConfig = None,
        repeat_count: int = 100,
        class_conditioner: Optional[ClassConditioner] = None,
        reweight_timesteps: bool = False,
        skip_validation: bool = False,
        num_time_buckets: int = 10,
        **kwargs,
    ):
        # Pass configs, not instantiated objects
        return cls(
            reconstructor=reconstructor,
            belief=belief,
            sampler_cfg=sampler,
            optimizer_cfg=optimizer,
            scheduler_cfg=scheduler,
            ema_cfg=ema,
            repeat_count=repeat_count,
            class_conditioner=class_conditioner,
            reweight_timesteps=reweight_timesteps,
            skip_validation=skip_validation,
            num_time_buckets=num_time_buckets,
            datamodule=datamodule,
            **kwargs,
        )

    def __init__(
        self,
        reconstructor: SampleReconstructor,
        belief: Belief,
        sampler_cfg: DictConfig,
        optimizer_cfg: DictConfig,
        scheduler_cfg: DictConfig,
        datamodule: pl.LightningDataModule,
        ema_cfg: DictConfig = None,
        repeat_count: int = 100,
        class_conditioner: Optional[ClassConditioner] = None,
        reweight_timesteps: bool = False,
        num_time_buckets: int = 10,
        skip_validation: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=("reconstructor","datamodule"))
        self.reconstructor = reconstructor
        self.ema_reconstructor = (
            instantiate(ema_cfg, model=reconstructor) if ema_cfg else None
        )

        self.belief = belief
        self.kl_by_bucket_sums: list[
            torch.Tensor
        ] = []  # List to store KL_by_bucket values during validation
        self.kl_by_bucket_counts: list[
            torch.Tensor
        ] = []  # List to store counts for KL_by_bucket
        self.kl_by_component_values: list[
            dict[str, float]
        ] = []  # List to store KL_by_component values during validation
        metrics = tm.MetricCollection(
            {
                "ELBO": tm.MeanMetric(),
                "log_likelihood": tm.MeanMetric(),
                "loss": tm.MeanMetric(),
                "KL": tm.MeanMetric(),
            }
        )
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")
        self.repeat_count = repeat_count
        self.class_conditioner = class_conditioner

        self.samplers: list[Sampler] = [
            instantiate(
                cfg,
                belief=belief,
                reconstructor=self.ema_reconstructor
                if self.ema_reconstructor
                else self.reconstructor,
                class_conditioner=class_conditioner,
            )
            for cfg in sampler_cfg.values()
        ]
        self.avg_kl_by_bucket: Optional[torch.Tensor] = None
        self.reweight_timesteps = reweight_timesteps
        self.num_time_buckets = num_time_buckets
        self.skip_validation = skip_validation
        self.datamodule = datamodule

    def setup(self, stage: str):
        self.belief.to(self.device)

    def estimate_kl(self, batch: TensorDict, disable_reweighting: bool = False):
        batch = batch.to(self.device)
        batch = batch.repeat_interleave(self.repeat_count, dim=0)
        batch_size = batch.batch_size
        timestep_weights = None
        t_buckets = None
        if (
            self.avg_kl_by_bucket is not None
            and self.reweight_timesteps
            and not disable_reweighting
        ):
            timestep_weights = self.avg_kl_by_bucket / self.avg_kl_by_bucket.sum() + 0.01
            timestep_weights /= timestep_weights.sum()
            # print("Using timestep weights for KL estimation:", timestep_weights.cpu().numpy())
            # Sample t from a categorical distribution based
            t_buckets = torch.distributions.Categorical(probs=timestep_weights).sample(
                batch_size
            )
            base_ts = t_buckets / timestep_weights.shape[0]
            t = (
                base_ts
                + torch.rand(batch_size, device=self.device) / timestep_weights.shape[0]
            )
        else:
            t = torch.rand(batch_size, device=self.device)
        beliefs = self.belief.sample_initial_belief(batch_size)
        if self.class_conditioner is not None:
            classifications = self.class_conditioner.classify_sample(batch)
            self.class_conditioner.write_sample_classifications_to_belief(
                classifications, beliefs
            )
        precision = self.belief.get_precision(t)
        self.belief.sample_belief_update_inplace(beliefs, batch, precision)
        beliefs["t"] = t
        reconstruction = self.reconstructor(beliefs)
        kl = self.belief.dKL_dt(beliefs, batch, reconstruction, t)  # .mean()
        if timestep_weights is not None:
            batch_weights = (
                1 / timestep_weights[t_buckets] / len(timestep_weights)
            )  # each bucket should have a weight of 1 / num_buckets but gets an extra weight of timestep_weights[t_buckets]
            kl = (kl * batch_weights).mean()
        else:
            kl = kl.mean()

        # Explicitly clear intermediate tensors
        del t, beliefs, precision, reconstruction, batch

        return kl

    def estimate_kl_by_timestep(self, batch: TensorDict, num_time_intervals: int = 10):
        batch = batch.to(self.device)
        batch = batch.repeat_interleave(self.repeat_count, dim=0)
        batch_size = batch.batch_size[0]

        num_repeats = int(math.floor(batch_size / num_time_intervals))
        t_buckets = torch.arange(num_time_intervals, device=self.device).repeat(
            num_repeats
        )
        # Randomly allocate rounding error for repeat count
        to_allocate_count = batch_size - num_repeats * num_time_intervals
        to_allocate = torch.randint(
            0, num_time_intervals, (to_allocate_count,), device=self.device
        )
        t_buckets = torch.cat([t_buckets, to_allocate], dim=0)
        # randomly permute t_buckets
        t_buckets = t_buckets[torch.randperm(t_buckets.shape[0])]
        # split t_buckets into num_time_intervals buckets
        t = (
            t_buckets / num_time_intervals
            + torch.rand(batch_size, device=self.device) / num_time_intervals
        )
        beliefs = self.belief.sample_initial_belief(batch_size)
        precision = self.belief.get_precision(t)
        self.belief.sample_belief_update_inplace(beliefs, batch, precision)
        beliefs["t"] = t
        if self.class_conditioner is not None:
            classifications = self.class_conditioner.classify_sample(batch)
            self.class_conditioner.write_sample_classifications_to_belief(
                classifications, beliefs
            )
        reconstruction = self.reconstructor(beliefs)
        kl = self.belief.dKL_dt(beliefs, batch, reconstruction, t)
        kl_by_bucket = torch.zeros(num_time_intervals, device=self.device)
        for i in range(num_time_intervals):
            kl_by_bucket[i] = kl[t_buckets == i].sum()
        bucket_counts = torch.bincount(t_buckets, minlength=num_time_intervals)
        return kl_by_bucket, bucket_counts

    def estimate_kl_by_belief_component(self, batch: TensorDict):
        if not isinstance(self.belief, ComposedBelief):
            return
        batch_size = batch.batch_size
        t = torch.rand(batch_size, device=self.device)
        beliefs = self.belief.sample_initial_belief(batch_size)
        precision = self.belief.get_precision(t)
        self.belief.sample_belief_update_inplace(beliefs, batch, precision)
        beliefs["t"] = t
        if self.class_conditioner is not None:
            classifications = self.class_conditioner.classify_sample(batch)
            self.class_conditioner.write_sample_classifications_to_belief(
                classifications, beliefs
            )
        reconstruction = self.reconstructor(beliefs)
        kl_by_component: dict[str, float] = {}
        for component in self.belief.beliefs:
            key = (
                component.key
                if hasattr(component, "key")
                else component.__class__.__name__
            )
            kl_by_component[key] = (
                component.dKL_dt(beliefs, batch, reconstruction, t).mean().item()
            )

        # Explicitly clear intermediate tensors
        del t, beliefs, precision, reconstruction, batch

        return kl_by_component

    def estimate_reconstruction_log_likelihood(self, batch: TensorDict):
        batch = batch.to(self.device)
        batch_size = batch.batch_size

        # Use no_grad for inference
        with torch.no_grad():
            t = torch.ones(batch_size, device=self.device)
            beliefs = self.belief.sample_initial_belief(batch_size)
            precision = self.belief.get_precision(t)
            self.belief.sample_belief_update_inplace(beliefs, batch, precision)
            beliefs["t"] = t
            if self.class_conditioner is not None:
                classifications = self.class_conditioner.classify_sample(batch)
                self.class_conditioner.write_sample_classifications_to_belief(
                    classifications, beliefs
                )
            likelihood = self.belief.log_prob(beliefs, batch).mean()

            # Explicitly clear intermediate tensors
            del t, beliefs, precision, batch

            return likelihood

    def training_step(self, batch_raw: dict[str, torch.Tensor], batch_idx):
        batch = dict_of_tensors_to_tensor_dict(batch_raw)
        kl = self.estimate_kl(batch)
        loss = kl / self.belief.get_dimentionality()
        self.log("train/loss", loss, batch_size=batch.batch_size[0])
        self.log("train/KL", kl, batch_size=batch.batch_size[0])

        # Clear the converted batch
        del batch

        return {"loss": loss}

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.ema_reconstructor:
            self.ema_reconstructor.update()

    def validation_step(self, batch_raw: dict[str, torch.Tensor], batch_idx):
        if self.skip_validation:
            return {
                "val/loss": 0,
                "val/log_likelihood": 0,
                "val/ELBO": 0,
                "val/KL": 0,
            }
        batch = dict_of_tensors_to_tensor_dict(batch_raw)
        kl = self.estimate_kl(batch, disable_reweighting=True)
        log_likelihood = self.estimate_reconstruction_log_likelihood(batch)
        elbo = log_likelihood - kl
        loss = kl / self.belief.get_dimentionality()
        validation_set_size = len(self.datamodule.val_dataloader().dataset)
        num_buckets = self.num_time_buckets
        min_samples_per_bucket = 20
        repeat_count = math.ceil(num_buckets * min_samples_per_bucket / validation_set_size)
        for _ in range(repeat_count):
            kl_by_bucket_sum, bucket_counts = self.estimate_kl_by_timestep(
                batch, num_time_intervals=self.num_time_buckets
            )
            self.kl_by_bucket_sums.append(kl_by_bucket_sum)  # Store KL_by_bucket values
            self.kl_by_bucket_counts.append(bucket_counts)
        kl_by_component = self.estimate_kl_by_belief_component(batch)
        self.kl_by_component_values.append(
            kl_by_component
        )  # Store KL_by_component values

        # Log metrics
        self.log("val/loss", loss, batch_size=batch.batch_size[0])
        self.log("val/KL", kl, batch_size=batch.batch_size[0])
        self.log("val/log_likelihood", log_likelihood, batch_size=batch.batch_size[0])
        self.log("val/ELBO", elbo, batch_size=batch.batch_size[0])

        # Clear the converted batch
        del batch

        return {
            "val/loss": loss,
            "val/log_likelihood": log_likelihood,
            "val/ELBO": elbo,
            "val/KL": kl,
        }

    def on_validation_epoch_start(self):
        self.kl_by_bucket_sums= []
        self.kl_by_bucket_counts = []
        self.kl_by_component_values = []

    def on_validation_epoch_end(self):
        if len(self.kl_by_bucket_sums) == 0:
            return

        self.avg_kl_by_bucket = (torch.stack(self.kl_by_bucket_sums).sum(dim=0) / torch.stack(self.kl_by_bucket_counts).sum(dim=0)).detach()
        # base_weights = avg_kl_by_bucket / avg_kl_by_bucket.sum()
        # self.timestep_weights = base_weights + 0.01  # Add a small constant to avoid zero weights

    def test_step(self, batch_raw: dict[str, torch.Tensor], batch_idx):
        batch = dict_of_tensors_to_tensor_dict(batch_raw)
        kl = self.estimate_kl(batch)
        log_likelihood = self.estimate_reconstruction_log_likelihood(batch)
        elbo = log_likelihood - kl

        # Log metrics
        self.log("test/loss", kl, batch_size=batch.batch_size[0])
        self.log("test/log_likelihood", log_likelihood, batch_size=batch.batch_size[0])
        self.log("test/ELBO", elbo, batch_size=batch.batch_size[0])

        # Clear the converted batch
        del batch

        return {
            "test/loss": kl,
            "test/log_likelihood": log_likelihood,
            "test/ELBO": elbo,
        }

    def sample(
        self,
        num_samples: int,
        batch_size: int = 512,
        belief_timesteps_to_save: list[float] = [],
        conditioner: Optional[Conditioner] = None,
    ):
        """
        Sample from the model.
        Args:
            num_samples (int): The number of samples to generate.
            batch_size (int): The number of samples to generate in each call to `sample_batch`.
            conditioner (Callable[[TensorDict],None]): A function that modifies the belief reconstruction in place for conditioning. May be used to fix some features of the reconstruction. If None, no conditioning is applied.
        Returns:
            TensorDict or tuple[list[TensorDict], Tensordict]: A batch of samples from the model. If `return_belief_series` is True, returns a list of beliefs at each time step and the final sample.
        """
        generated_samples: dict[str, TensorDict] = {}
        generated_beliefs: dict[str, list[TensorDict]] = {}
        for sampler in self.samplers:
            samples, beliefs = sampler.sample(
                num_samples=num_samples,
                batch_size=batch_size,
                belief_timesteps_to_save=belief_timesteps_to_save,
                conditioner=conditioner,
            )
            generated_samples[sampler.name] = samples
            generated_beliefs[sampler.name] = beliefs

        return generated_samples, generated_beliefs

    def sample_encoding_belief_series(
        self,
        ground_truth: TensorDict,
        time_discretization_steps: int = 100,
        belief_timesteps_to_save: list[float] = [],
    ):
        """
        Sample from the denoising process observing a given ground truth.
        Args:
            ground_truth (TensorDict): The ground truth to observe.
            time_discretization_steps (int): The number of time steps to discretize the time interval [0, 1].
        Returns:
            final_sample (TensorDict): The final sample from the denoising process.
            belief_series (list[TensorDict]): A list of beliefs at each time step.
        """
        with torch.no_grad():
            belief_series: list[TensorDict] = []
            belief = self.belief.sample_initial_belief(ground_truth.batch_size)
            belief["t"] = torch.zeros(ground_truth.batch_size, device=self.device)
            if self.class_conditioner is not None:
                classifications = self.class_conditioner.classify_sample(ground_truth)
                self.class_conditioner.write_sample_classifications_to_belief(
                    classifications, belief
                )
            belief_timesteps_to_save = sorted(belief_timesteps_to_save)
            belief_timesteps_save_idx = 0
            if 0 in belief_timesteps_to_save:
                # If the first timestep to save is 0, we save the initial belief
                belief_series.append(belief.clone())
                belief_timesteps_save_idx += 1

            for i in range(time_discretization_steps):
                dt = 1 / time_discretization_steps
                t = torch.full(ground_truth.batch_size, i * dt, device=self.device)
                belief["t"] = t
                precision_increment = self.belief.get_dprecision_dt(t) * dt
                self.belief.sample_belief_update_inplace(
                    belief, ground_truth, precision_increment
                )

                while (
                    belief_timesteps_save_idx < len(belief_timesteps_to_save)
                    and i * dt
                    <= belief_timesteps_to_save[belief_timesteps_save_idx]
                    <= i * dt + dt
                ):
                    # Add beliefs to the series for all timesteps to save in [t, t + dt]. Normally this would only include one timestep but the while loop ensures that the number of saved beliefs is equal to the number of timesteps to save.
                    belief_series.append(belief.clone())
                    belief_timesteps_save_idx += 1

                # Clear intermediate tensors
                del t, precision_increment

            final_sample = self.belief.sample_from_belief(belief)
            del belief

            return final_sample, belief_series

    def configure_optimizers(self):
        optimizer = instantiate(
            self.hparams.optimizer_cfg, params=self.reconstructor.parameters()
        )
        scheduler = instantiate(self.hparams.scheduler_cfg, optimizer=optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
                "monitor": "train/loss",
            },
        }

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.belief.to(*args, **kwargs)
        self.reconstructor.to(*args, **kwargs)
        for sampler in self.samplers:
            sampler.to(*args, **kwargs)
        return self
