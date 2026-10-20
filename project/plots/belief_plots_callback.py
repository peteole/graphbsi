from typing import Optional, Protocol

import lightning.pytorch as pl
import torch
import wandb
import wandb.data_types
from hydra.utils import instantiate
from omegaconf import DictConfig
from tensordict import TensorDict

from project.tasks import ELBO
from project.utils.tensor_utils import dict_of_tensors_to_tensor_dict


class PlotSamplesFunc(Protocol):
    def __call__(
        self,
        samples: TensorDict,
        captions: Optional[list[str]] = None,
        title: Optional[str] = None,
    ) -> wandb.data_types.Media:
        """
        Plot a set of samples, with optionally a caption for each sample and an overall title.
        """
        ...


class PlotBeliefSeriesFunc(Protocol):
    def __call__(
        self,
        belief_series: list[TensorDict],
        times: list[float],
        title: Optional[str] = None,
    ) -> wandb.data_types.Media:
        """
        Plot a set of belief series at different times, with an overall title.
        The belief series are a list of TensorDicts, with each element representing a set of samples at a given time.
        The times are a list of floats, with each element representing the time at which the corresponding belief series was sampled.
        Args:
            belief_series: A list of TensorDicts, with each element representing a set of samples at a given time. All must have the same shape.
            times: A list of floats, with each element representing the time at which the corresponding belief series was sampled. Same length as belief_series.
            title: An optional overall title for the plot.
        Returns:
            A wandb.data_types.Media object representing the plot.
        """
        ...


class BeliefPlotsCallback(pl.Callback):
    @classmethod
    def from_config(
        cls,
        datamodule: pl.LightningDataModule,
        samples_plotter: DictConfig,
        belief_series_plotter: DictConfig,
        num_timesteps_to_plot: int = 10,
        num_samples_to_plot: int = 10,
        num_belief_series_to_plot: int = 3,
        batch_size: int = 64,
        num_time_discretization_steps: int = 100,
        **kwargs,
    ):
        plot_samples = instantiate(samples_plotter, datamodule=datamodule)
        plot_belief_series = instantiate(belief_series_plotter, datamodule=datamodule)
        return cls(
            datamodule=datamodule,
            plot_samples=plot_samples,
            plot_belief_series=plot_belief_series,
            num_timesteps_to_plot=num_timesteps_to_plot,
            num_time_discretization_steps=num_time_discretization_steps,
            num_samples_to_plot=num_samples_to_plot,
            num_belief_series_to_plot=num_belief_series_to_plot,
            batch_size=batch_size,
        )

    def __init__(
        self,
        datamodule: pl.LightningDataModule,
        plot_samples: PlotSamplesFunc,
        plot_belief_series: PlotBeliefSeriesFunc,
        num_timesteps_to_plot: int = 10,
        num_time_discretization_steps: int = 100,
        num_samples_to_plot: int = 10,
        num_belief_series_to_plot: int = 3,
        batch_size: int = 64,
    ):
        super().__init__()
        self.datamodule = datamodule
        self.plot_samples = plot_samples
        self.plot_belief_series = plot_belief_series
        self.num_timesteps_to_plot = num_timesteps_to_plot
        self.num_time_discretization_steps = num_time_discretization_steps
        self.num_samples_to_plot = num_samples_to_plot
        self.num_belief_series_to_plot = num_belief_series_to_plot
        self.batch_size = batch_size

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        # Plot samples from the validation set
        samples = self._get_validation_set_samples(self.num_samples_to_plot)
        samples_plots = self.plot_samples(samples)
        wandb.log({"setup/validation_set_samples": samples_plots})

        if not isinstance(pl_module, ELBO):
            return
        samples = self._get_validation_set_samples(self.num_belief_series_to_plot)
        samples = samples.to(pl_module.device)
        print("device for plotting encoding belief series:", pl_module.device)
        times = torch.linspace(0, 1, self.num_timesteps_to_plot)
        generated_samples, encoding_belief_series = (
            pl_module.sample_encoding_belief_series(
                samples,
                time_discretization_steps=self.num_time_discretization_steps,
                belief_timesteps_to_save=times.tolist(),
            )
        )
        encoding_belief_series_plots = self.plot_belief_series(
            encoding_belief_series, times.tolist()
        )
        wandb.log({"setup/encoding_belief_series": encoding_belief_series_plots})

    def on_validation_epoch_end(self, trainer, pl_module):
        if not isinstance(pl_module, ELBO):
            return
        times = torch.linspace(0, 1, self.num_timesteps_to_plot)
        generated_samples, _ = pl_module.sample(
            num_samples=self.num_samples_to_plot,
            batch_size=self.batch_size,
            # belief_timesteps_to_save=times.tolist(),
        )
        generated_samples_bs, belief_series = pl_module.sample(
            num_samples=self.num_belief_series_to_plot,
            batch_size=self.batch_size,
            belief_timesteps_to_save=times.tolist(),
        )

        for sampler in generated_samples.keys():
            # plot generated samples
            print(f"plotting generated samples for {sampler}")
            generated_samples_plots = self.plot_samples(generated_samples[sampler])
            wandb.log({f"val/generated_samples_{sampler}": generated_samples_plots})

            # plot belief series and its reconstructions
            print(f"plotting belief series for {sampler}")
            belief_series_plots = self.plot_belief_series(
                belief_series[sampler], times.tolist()
            )
            wandb.log({f"val/belief_series_{sampler}": belief_series_plots})
            reconstructions_plots = self._plot_reconstructions(
                generated_samples_bs[sampler],
                belief_series[sampler],
                pl_module,
            )
            wandb.log(
                {f"val/belief_series_reconstructions_{sampler}": reconstructions_plots}
            )

        # plot encoding reconstructions
        print("plotting encoding reconstructions")
        validation_samples = self._get_validation_set_samples(
            self.num_belief_series_to_plot
        )
        validation_samples = validation_samples.to(pl_module.device)
        generated_samples, encoding_belief_series_full = (
            pl_module.sample_encoding_belief_series(
                validation_samples,
                time_discretization_steps=self.num_time_discretization_steps,
                belief_timesteps_to_save=times.tolist(),
            )
        )
        reconstructions_plots = self._plot_reconstructions(
            validation_samples, encoding_belief_series_full, pl_module
        )
        wandb.log({"val/encoding_reconstructions": reconstructions_plots})

        print("done")

    def _plot_reconstructions(
        self,
        ground_truth_samples: TensorDict,
        belief_series: list[TensorDict],
        pl_module: ELBO,
    ) -> wandb.data_types.Media:
        ground_truth_samples = pl_module.belief.expand_sample(ground_truth_samples)
        times = torch.linspace(0, 1, self.num_timesteps_to_plot)
        num_belief_series = ground_truth_samples.batch_size[0]
        for belief in belief_series:
            assert belief.batch_size[0] == num_belief_series
        concat_encoding_belief_series = torch.cat(belief_series, dim=0)
        encoding_reconstructions = pl_module.reconstructor(concat_encoding_belief_series)
        samples_to_plot = []
        captions = []
        for i in range(num_belief_series):
            samples_to_plot.append(ground_truth_samples[i])
            captions.append(f"Sample {i}")
            for j in range(self.num_timesteps_to_plot):
                samples_to_plot.append(
                    encoding_reconstructions[num_belief_series * j + i]
                )
                captions.append(f"Rec. {i} @ t={times[j]:.2f}")
        cat_encoding_belief_series: TensorDict = torch.stack(samples_to_plot, dim=0)
        encoding_reconstructions_plots = self.plot_samples(
            cat_encoding_belief_series, captions
        )
        return encoding_reconstructions_plots

    def _get_validation_set_samples(self, num_samples: int) -> TensorDict:
        batches = []
        current_num_samples = 0
        for batch in self.datamodule.val_dataloader():
            batch = dict_of_tensors_to_tensor_dict(batch)
            batch_size = batch.batch_size[0]
            to_append = batch[: num_samples - current_num_samples]
            batches.append(to_append)
            current_num_samples += batch_size
            if current_num_samples >= num_samples:
                break
        return torch.cat(batches, dim=0)

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not isinstance(pl_module, ELBO):
            return
        generated_samples, _ = pl_module.sample(
            num_samples=self.num_samples_to_plot,
            batch_size=self.batch_size,
        )
        for sampler, samples in generated_samples.items():
            generated_samples_plots = self.plot_samples(samples)
            wandb.log({f"final/generated_samples_{sampler}": generated_samples_plots})
