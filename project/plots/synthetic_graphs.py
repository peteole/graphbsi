from typing import Literal

import lightning.pytorch as pl
import matplotlib.pyplot as plt
import networkx as nx
import torch
import wandb
from synthetic_graph_benchmarks import (
    benchmark_planar_results,
    benchmark_sbm_results,
    benchmark_tree_results,
)
from tensordict import TensorDict
from wandb.sdk.data_types.base_types.media import Media

from project.data.synthetic_graphs import SyntheticGraphDataModule, sample_to_graph
from project.plots.belief_plots_callback import PlotBeliefSeriesFunc, PlotSamplesFunc
from project.samplers.sampler import Sampler
from project.tasks import ELBO
from project.utils.metrics_capture import record_many


def _force_serial_sbm_validity() -> None:
    """Evaluate SBM validity serially.

    `eval_acc_sbm_graph` fans the per-graph check out over a ThreadPoolExecutor,
    but that check runs graph-tool's blockmodel MCMC, whose C++ core is not safe
    to re-enter from several Python threads: it segfaults partway through
    (SIGSEGV in `_multilevel_mcmc_sweep_dispatch`). Serial evaluation of a couple
    hundred graphs costs minutes and is the difference between a number and a
    crash.

    No-op when graph-tool is absent -- there the pure-Python spectral-clustering
    fallback runs instead, which is thread-safe (but not the metric the paper
    reports; see README).
    """
    try:
        from synthetic_graph_benchmarks import spectre_utils
    except ImportError:  # pragma: no cover - benchmark package always present in practice
        return

    original = spectre_utils.eval_acc_sbm_graph
    if getattr(original, "_graphbsi_serial", False):
        return

    def serial_eval_acc_sbm_graph(*args, **kwargs):
        kwargs["is_parallel"] = False
        return original(*args, **kwargs)

    serial_eval_acc_sbm_graph._graphbsi_serial = True  # type: ignore[attr-defined]
    spectre_utils.eval_acc_sbm_graph = serial_eval_acc_sbm_graph


_force_serial_sbm_validity()


def generate_synthetic_graphs(
    sampler: Sampler,
    num_samples: int,
    batch_size: int = 200,
):
    samples, _ = sampler.sample(num_samples, batch_size=batch_size)
    graphs = [sample_to_graph(sample) for sample in samples]
    return graphs


class SyntheticGraphScoresCallback(pl.Callback):
    def __init__(
        self,
        datamodule: SyntheticGraphDataModule,
        ds_type: Literal["planar", "sbm", "tree"],
        num_samples: int = 200,
        batch_size: int = 200,
        metric_batch_size: int = 40,
        **kwargs,
    ):
        super().__init__()
        self.datamodule = datamodule
        self.ds_type = ds_type
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.metric_batch_size = metric_batch_size

    def on_fit_start(self, trainer, pl_module):
        assert self.datamodule.train_samples is not None, (
            "train_samples must be set in the SyntheticGraphDataModule"
        )
        assert isinstance(pl_module, ELBO), (
            "This callback is only compatible with ELBO tasks"
        )
        train_graphs = [sample_to_graph(sample) for sample in self.datamodule.train_samples]
        if self.ds_type == "planar":
            res = benchmark_planar_results(train_graphs)
        elif self.ds_type == "sbm":
            res = benchmark_sbm_results(train_graphs)
        elif self.ds_type == "tree":
            res = benchmark_tree_results(train_graphs)
        else:
            raise ValueError(f"Unknown dataset type: {self.ds_type}")
        for key, value in res.items():
            wandb.log({f"{self.ds_type}_train/{key}": value})

    def on_validation_epoch_end(self, trainer, pl_module):
        assert isinstance(pl_module, ELBO), (
            "This callback is only compatible with ELBO tasks"
        )
        if pl_module.trainer.current_epoch == 0:
            print("Skipping validation at epoch 0")
            return
        for sampler in pl_module.samplers:
            graphs = generate_synthetic_graphs(
                sampler, num_samples=self.num_samples, batch_size=self.batch_size
            )
            # convert to json
            graphs_json = [nx.node_link_data(g) for g in graphs]
            # Log graphs as JSON artifacts
            wandb.log({f"{sampler.name[:100]}/graphs": wandb.Table(
                columns=["graph_id", "graph_data"], 
                data=[[i, graph_json] for i, graph_json in enumerate(graphs_json)]
            )})
            batch_metrics: list[dict[str, float]] = []
            for i in range(0, self.num_samples, self.metric_batch_size):
                batch_graphs = graphs[i : i + self.metric_batch_size]
                if self.ds_type == "planar":
                    res = benchmark_planar_results(batch_graphs)
                elif self.ds_type == "sbm":
                    res = benchmark_sbm_results(batch_graphs)
                elif self.ds_type == "tree":
                    res = benchmark_tree_results(batch_graphs)
                else:
                    raise ValueError(f"Unknown dataset type: {self.ds_type}")
                batch_metrics.append(res)
            
            # average metrics over batches
            mean_res = {key: sum(d[key] for d in batch_metrics) / len(batch_metrics) for key in batch_metrics[0]}
            std_res = {key: (sum((d[key] - mean_res[key])**2 for d in batch_metrics) / len(batch_metrics))**0.5 for key in batch_metrics[0]}
            for key in mean_res:
                print(f"{key}: {mean_res[key]:.5f} ± {std_res[key]:.4f}")
                payload = {
                    f"{sampler.name}/{key}": mean_res[key],
                    f"{sampler.name}_std/{key}": std_res[key],
                }
                record_many(payload)
                wandb.log({f"{sampler.name}/{key}": mean_res[key]})
                wandb.log({f"{sampler.name}_std/{key}": std_res[key]})



class PlotGraphSamples(PlotSamplesFunc):
    def __init__(self, num_cols: int = 5, **kwargs):
        super().__init__(**kwargs)
        self.num_cols = num_cols
    
    def __call__(self, samples: TensorDict, captions: list[str] | None = None, title: str | None = None):
        graphs = [sample_to_graph(sample) for sample in samples]
        fig, axes = plt.subplots(
            nrows=(len(graphs) + self.num_cols - 1) // self.num_cols,
            ncols=self.num_cols,
            figsize=(15, 3 * ((len(graphs) + self.num_cols - 1) // self.num_cols)),
        )
        axes = axes.flatten() if len(graphs) > 1 else [axes]
        for ax, graph in zip(axes, graphs):
            nx.draw(graph, ax=ax, with_labels=False, node_size=50, font_size=8)
            if captions:
                ax.set_title(captions.pop(0))
            else:
                ax.set_title("Graph")
        for ax in axes[len(graphs):]:
            ax.axis("off")
        if title:
            plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        media = wandb.Image(fig, caption=title) if title else wandb.Image(fig)
        plt.close(fig)
        return media


class PlotGraphBeliefSeries(PlotBeliefSeriesFunc):
    def __init__(self,  **kwargs):
        pass
    def __call__(self, belief_series: list[TensorDict], times: list[float], title: str | None = None) -> Media:
        num_cols = len(times)
        num_rows = belief_series[0].batch_size[0]  # Assuming batch size is the same for all samples
        fig, axes = plt.subplots(
            nrows=num_rows,
            ncols=num_cols,
            figsize=(num_cols * 3, num_rows * 3),
        )
        if num_rows == 1:
            axes = [axes]
        for j, batch_of_samples in enumerate(belief_series):
            for i, sample in enumerate(batch_of_samples):
                ax = axes[i][j] if num_rows > 1 else axes[j]
                adjacency_matrix: torch.Tensor = sample["E"].cpu()
                if adjacency_matrix.ndim == 3:
                    if adjacency_matrix.shape[-1] ==2:
                        adjacency_matrix = torch.softmax(adjacency_matrix,dim=-1)[..., 1]
                    else:
                        adjacency_matrix = adjacency_matrix.argmax(-1)
                # nx.draw(
                #     nx.from_numpy_array(adjacency_matrix),
                #     ax=ax,
                #     with_labels=True,
                #     node_size=50,
                #     font_size=8,
                # )
                ax.matshow(adjacency_matrix.numpy(), cmap='viridis')
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(f"Time: {times[j]:.2f}")
        if title:
            plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        media = wandb.Image(fig, caption=title) if title else wandb.Image(fig)
        plt.close(fig)
        return media