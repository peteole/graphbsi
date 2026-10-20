import pickle
import re
from typing import Dict, Optional, Tuple, Union

import lightning.pytorch as pl
import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from tqdm import tqdm

from project.samplers.sampler import Sampler

matplotlib.use("Agg")
import hashlib

import networkx as nx
import numpy as np
import torch
import wandb
from molecule_benchmarks import Benchmarker, SmilesDataset
from rdkit import Chem, RDLogger
from tensordict import TensorDict
from torch.nn.functional import softmax
from wandb import Image
from wandb.sdk.data_types.base_types.media import Media

from project.data.molecule import MoleculeDataModule
from project.plots.belief_plots_callback import PlotBeliefSeriesFunc, PlotSamplesFunc
from project.tasks import ELBO
from project.utils.colors import generate_distinct_colors
from project.utils.metrics_capture import record_many

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


def plot_molecule_distribution(
    data: Union[TensorDict, Dict[str, np.ndarray]],  # Contains X, E, mask tensors
    datamodule: MoleculeDataModule,
    captions: Optional[list[str]] = None,
    title: Optional[str] = None,
    samples_per_row: int = 5,
    node_radius: float = 0.08,
    edge_radius: float = 0.04,
    edge_width: float = 2.0,
    subplot_size: Tuple[float, float] = (4, 4),  # Size of each subplot in inches,
    plot_smiles: bool = False,
) -> plt.Figure:
    """
    Visualize a batch of molecule distributions using a grid of graphs with pie charts.

    Args:
        data: Either a TensorDict or dictionary containing:
            - X: Tensor of shape (batch_size, num_atoms, num_atom_types) containing atom type probabilities
            - E: Tensor of shape (batch_size, num_atoms, num_atoms, num_bond_types) containing bond type probabilities
            - mask: Tensor of shape (batch_size, num_atoms) containing mask of atoms to include
        atom_type_names: Dictionary mapping atom names to their indices
        bond_type_names: Dictionary mapping bond names to their indices
        samples_per_row: Number of samples to display per row in the grid
        node_radius: Pie chart radius for nodes
        edge_radius: Pie chart radius for edges
        edge_width: Width of edges
        subplot_size: Size of each subplot in inches (width, height)
    """
    # atom_type_names = datamodule.get_atom_type_names()  # maps atom name to index
    # bond_type_names = datamodule.get_bond_type_names()  # maps bond name to index
    # Convert TensorDict to numpy if needed
    if isinstance(data, TensorDict):
        X: np.ndarray = data["X"].detach().cpu().numpy()
        E: np.ndarray = data["E"].detach().cpu().numpy()
        mask = data["mask"].detach().cpu().numpy()
    else:
        X = data["X"].detach().cpu().numpy()
        E = data["E"].detach().cpu().numpy()
        mask = data["mask"].detach().cpu().numpy()
    batch_size, max_num_atoms, num_atom_types = X.shape
    num_rows = (batch_size + samples_per_row - 1) // samples_per_row  # Ceiling division
    assert E.ndim == 4, (
        "E must be a 4D tensor of shape (batch_size, num_atoms, num_atoms, num_bond_types)"
    )
    num_bond_types = E.shape[-1]
    assert E.shape[0] == batch_size, (
        f"Batch size of E ({E.shape[0]}) must match batch size of X ({batch_size})"
    )
    assert E.shape[1] == max_num_atoms and E.shape[2] == max_num_atoms, (
        f"Number of atoms in E ({E.shape[1]}, {E.shape[2]}) must match number of atoms in X ({max_num_atoms})"
    )
    smiles: list[str] = []
    for i in range(batch_size):
        mol = datamodule.molecular_sample_to_molecule(
            {
                "X": torch.tensor(X[i]),
                "E": torch.tensor(E[i]),
                "mask": torch.tensor(mask[i]),
            }
        )
        # mol = make_molecule(torch.tensor(X[i]), torch.tensor(E[i]), torch.tensor(mask[i]))

        if mol is not None and is_valid_molecule(mol):
            smiles.append(Chem.MolToSmiles(mol))
        else:
            smiles.append("Invalid molecule")

    num_atom_types_wo_none = num_atom_types - 1  # Exclude not present type (index 0)
    num_bond_types_wo_none = num_bond_types - 1  # Exclude no-bond type (index 0)

    atom_colors = generate_distinct_colors(num_atom_types_wo_none, seed=1)
    bond_colors = generate_distinct_colors(num_bond_types_wo_none, seed=2)

    # Calculate figure size based on grid layout
    fig_width = samples_per_row * subplot_size[0]
    fig_height = num_rows * subplot_size[1]

    # Create figure with grid of subplots
    fig, axes = plt.subplots(
        num_rows, samples_per_row, figsize=(fig_width, fig_height), squeeze=False
    )
    axes = axes.flatten()

    # Get a colormap for the pie charts
    if captions is not None:
        assert len(captions) == batch_size, "Number of captions must match batch size"
    else:
        captions = [f"Sample {i + 1}" for i in range(batch_size)]
    if plot_smiles:
        for i in range(batch_size):
            captions[i] = f"{captions[i]}\n{smiles[i]}"

    # Process each molecule in the batch
    for idx in range(batch_size):
        ax = axes[idx]
        curr_X = X[idx]
        curr_E = E[idx]
        curr_mask = mask[idx].astype(bool)

        # Apply mask
        curr_X = curr_X[curr_mask]
        curr_E = curr_E[curr_mask][:, curr_mask]

        # Create graph based on most likely molecule
        G = nx.Graph()
        num_atoms = curr_X.shape[0]

        # Add nodes
        for i in range(num_atoms):
            G.add_node(i)

        # Add edges based on most likely bonds (excluding no-bond type 0)
        for i in range(num_atoms):
            for j in range(i + 1, num_atoms):
                bond_prob = 1 - curr_E[i, j, 0]
                if bond_prob > 0.1:  # Only show edges with significant probability
                    G.add_edge(i, j, weight=bond_prob)

        # Use a more compact layout
        pos = nx.kamada_kawai_layout(G)

        # Normalize positions
        if len(pos) > 0:
            pos_arr = np.array(list(pos.values()))
            min_x, min_y = pos_arr.min(axis=0)
            max_x, max_y = pos_arr.max(axis=0)
            scale_x = max_x - min_x
            scale_y = max_y - min_y
            for k in pos:
                x, y = pos[k]
                if np.isclose(scale_x, 0):
                    norm_x = 0.5
                else:
                    norm_x = (x - min_x) / (scale_x + 1e-8)
                if np.isclose(scale_y, 0):
                    norm_y = 0.5
                else:
                    norm_y = (y - min_y) / (scale_y + 1e-8)
                pos[k] = np.array([norm_x, norm_y])

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # Draw node pies
        for i in G.nodes():
            x, y = pos[i]
            atom_probs = curr_X[i][1:]
            assert len(atom_probs) == num_atom_types_wo_none, (
                f"Expected {num_atom_types_wo_none} atom probabilities, got {len(atom_probs)}"
            )
            if atom_probs.sum() == 0:
                print(f"Atom {i} has no probability")
                continue
            ax.pie(
                atom_probs,
                center=(x, y),
                radius=node_radius,
                colors=[atom_colors[j] for j in range(len(atom_probs))],
                wedgeprops={"alpha": 0.85},
                frame=False,
            )

        # Draw edges
        for i, j in G.edges():
            edge_probs = curr_E[i, j]
            bond_prob = 1 - edge_probs[0]
            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=[(i, j)],
                edge_color="black",
                alpha=bond_prob,
                width=edge_width,
                ax=ax,
            )

        # Draw edge pies
        for i, j in G.edges():
            edge_probs = curr_E[i, j, 1:]
            bond_prob = sum(edge_probs)  # Sum over bond types excluding no-bond type
            assert len(edge_probs) == num_bond_types_wo_none, (
                f"Expected {num_bond_types_wo_none} bond probabilities, got {len(edge_probs)}"
            )
            if bond_prob > 0.1:
                x1, y1 = pos[i]
                x2, y2 = pos[j]
                mid_x = (x1 + x2) / 2
                mid_y = (y1 + y2) / 2
                ax.pie(
                    edge_probs,
                    center=(mid_x, mid_y),
                    radius=edge_radius,
                    colors=[bond_colors[k] for k in range(len(edge_probs))],
                    wedgeprops={"alpha": 0.85},
                    frame=False,
                )

        ax.set_axis_off()
        ax.set_title(captions[idx])

    # Hide unused subplots
    for idx in range(batch_size, len(axes)):
        axes[idx].set_visible(False)

    # Create legend elements (but do not add to axes)
    atom_legend_elements = []
    bond_legend_elements = []

    used_atom_indices = range(1, num_atom_types_wo_none + 1)
    for idx in used_atom_indices:
        name = datamodule.get_atom_type_names().get(idx, f"Atom {idx}")
        color = atom_colors[idx - 1]  # idx starts from 1, colors from 0

        atom_legend_elements.append(
            mpatches.Patch(facecolor=color, label=f"Atom: {name}")
        )

    used_bond_indices = range(1, num_bond_types_wo_none + 1)
    for idx in used_bond_indices:
        name = datamodule.get_bond_type_names().get(idx, f"Bond {idx}")
        color = bond_colors[idx - 1]  # idx starts from 1, colors from 0
        bond_legend_elements.append(
            mpatches.Patch(facecolor=color, label=f"Bond: {name}")
        )

    # Combine legends and add to the bottom of the figure
    all_legend_elements = atom_legend_elements + bond_legend_elements
    legend_labels = [p.get_label() for p in all_legend_elements]
    fig.legend(
        handles=all_legend_elements,
        labels=legend_labels,
        loc="lower center",
        # bbox_to_anchor=(0.5, -0.02),
        ncol=min(5, len(all_legend_elements)),
        fontsize=8,
        title="Atom Types / Bond Types",
        title_fontsize=9,
        frameon=False,
    )
    if title is not None:
        fig.suptitle(title, fontsize=12, fontweight="bold")
    # Adjust layout
    # plt.tight_layout()
    # plt.subplots_adjust(bottom=0.13)

    return fig


class MoleculeSamplesPlotter(PlotSamplesFunc):
    @classmethod
    def from_config(cls, datamodule: MoleculeDataModule, **kwargs):
        assert isinstance(datamodule, MoleculeDataModule), (
            "datamodule must be a MoleculeDataModule"
        )
        return cls(datamodule)

    def __init__(
        self,
        datamodule: MoleculeDataModule,
    ):
        self.datamodule = datamodule

    def __call__(
        self,
        samples: TensorDict,
        captions: list[str] | None = None,
        title: str | None = None,
    ) -> Media:
        fig = plot_molecule_distribution(
            samples,
            self.datamodule,
            captions,
            title,
            plot_smiles=True,
        )
        img = Image(fig)
        plt.close(fig)
        return img


class MoleculeBeliefSeriesPlotter(PlotBeliefSeriesFunc):
    @classmethod
    def from_config(cls, datamodule: MoleculeDataModule, **kwargs):
        assert isinstance(datamodule, MoleculeDataModule), (
            "datamodule must be a MoleculeDataModule"
        )
        return cls(datamodule)

    def __init__(self, datamodule: MoleculeDataModule):
        self.datamodule = datamodule

    def __call__(
        self,
        belief_series: list[TensorDict],
        times: list[float],
        title: str | None = None,
    ) -> Media:
        num_items_per_belief_series = belief_series[0].batch_size[0]
        reordered_belief_series = []
        captions = []
        for i in range(num_items_per_belief_series):
            for j in range(len(belief_series)):
                reordered_belief_series.append(belief_series[j][i])
                captions.append(f"t={times[j]:.2f}")
        cat_belief_series: TensorDict = torch.stack(reordered_belief_series, dim=0)
        cat_belief_series["X"] = softmax(cat_belief_series["X"], dim=-1)
        cat_belief_series["E"] = softmax(cat_belief_series["E"], dim=-1)
        cat_belief_series = cat_belief_series.to("cpu")
        fig = plot_molecule_distribution(
            cat_belief_series,
            self.datamodule,
            captions,
            title,
            samples_per_row=len(belief_series),
        )
        img = Image(fig)
        plt.close(fig)
        return img


def is_valid_molecule(mol: Chem.Mol) -> bool:
    try:
        Chem.SanitizeMol(mol)
        Chem.Kekulize(mol)
        return True
    except Exception:
        return False


def generate_molecules(
    sampler: Sampler, datamodule: MoleculeDataModule, num_molecules: int, batch_size: int
) -> list[str | None]:
    smiles: list[str | None] = []
    with tqdm(total=num_molecules, desc="Generating molecules") as pbar:
        while len(smiles) < num_molecules:
            batch, _belief_series = sampler.sample_batch(batch_size=batch_size)
            for mol in batch:
                mol = datamodule.molecular_sample_to_molecule(mol)
                if mol is not None:
                    smiles.append(Chem.MolToSmiles(mol))
                else:
                    smiles.append(None)
                # pbar.update(1)
            pbar.update(len(batch))
            del batch, _belief_series  # Free memory
    return smiles


class MoleculeScorePlotter(pl.Callback):
    @classmethod
    def from_config(
        cls,
        datamodule: MoleculeDataModule,
        num_samples: int = 10000,
        batch_size: int = 256,
        rebuild_benchmarker: bool = False,
        **kwargs,
    ):
        assert isinstance(datamodule, MoleculeDataModule), (
            "datamodule must be a MoleculeDataModule"
        )

        return cls(datamodule, num_samples, batch_size, rebuild_benchmarker)

    def __init__(
        self,
        datamodule: MoleculeDataModule,
        num_samples: int = 10000,
        batch_size: int = 256,
        rebuild_benchmarker: bool = False,
    ):
        """
        Args:
            rebuild_benchmarker: build the benchmark reference even if it is
                cached. Building it draws random numbers, so a seeded run only
                reproduces a run that built it as well.
        """
        self.datamodule = datamodule
        benchmarker_hash = MoleculeScorePlotter.benchmarker_hash(
            datamodule.smiles_dataset, num_samples
        )
        cache_path = datamodule.cache_dir / f"benchmarker_{benchmarker_hash}.pkl"
        if rebuild_benchmarker:
            print("Building a new benchmarker (rebuild_benchmarker=true)")
            self.benchmarker = Benchmarker(
                dataset=datamodule.smiles_dataset, num_samples_to_generate=num_samples
            )
        elif cache_path.exists():
            print(f"Loading cached benchmarker from {cache_path}")
            with open(cache_path, "rb") as f:
                self.benchmarker = pickle.load(f)
        else:
            print(f"Creating new benchmarker and saving to {cache_path}")
            self.benchmarker = Benchmarker(
                dataset=datamodule.smiles_dataset, num_samples_to_generate=num_samples
            )
            with open(cache_path, "wb") as f:
                pickle.dump(self.benchmarker, f)
        self.batch_size = batch_size

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if not isinstance(pl_module, ELBO):
            print("MoleculeScorePlotter is only compatible with ELBO")
            return
        print("plotting molecule score")
        if pl_module.trainer.current_epoch == 0:
            print("Skipping validation at epoch 0")
            return

        for sampler in pl_module.samplers:
            sampler_name = sampler.name
            print(f"Running benchmark for sampler: {sampler_name}")
            # molecule_generator = BSIMoleculeGenerationModel(
            #     sampler=sampler,
            #     datamodule=self.datamodule,
            #     batch_size=self.batch_size,
            # )
            generated_smiles = generate_molecules(
                sampler,
                self.datamodule,
                num_molecules=self.benchmarker.num_samples_to_generate,
                batch_size=self.batch_size,
            )
            artifact_name = re.sub(
                r"[^a-zA-Z0-9._-]",
                "_",
                f"generated_smiles_{sampler_name}_epoch_{pl_module.trainer.current_epoch}",
            )
            print(f"Logging generated SMILES to artifact: {artifact_name}")
            smiles_artifact = wandb.Artifact(
                artifact_name,
                type="generated_smiles",
                description=f"Generated SMILES from {sampler_name} sampler",
            )
            smiles_table = wandb.Table(
                data=[[smiles] for smiles in generated_smiles],
                columns=["smiles"],
            )
            smiles_artifact.add(
                smiles_table,
                "generated_smiles",
            )
            wandb.log({f"tables/generated_smiles/{sampler_name}": smiles_table})
            wandb.log_artifact(smiles_artifact)
            self.benchmarker.device = str(pl_module.device)
            scores = self.benchmarker.benchmark(generated_smiles)
            if scores is None:
                print(f"No valid molecules generated by {sampler_name}")
                continue
            print(f"Scores for {sampler_name}: {scores}")
            for metric, score in flatten_dict(scores).items():
                print(f"{sampler_name} - {metric}: {score:.4f}")
                payload = {f"val/{sampler_name}/{metric}": score}
                record_many(payload)
                wandb.log(payload)

        print("done")

    @staticmethod
    def benchmarker_hash(dataset: SmilesDataset, num_samples_to_generate: int) -> str:
        """
        Generate a unique hash for the benchmarker configuration.

        Args:
            dataset: The dataset used for benchmarking.
            num_samples_to_generate: Number of samples to generate.

        Returns:
            A unique hash string representing the benchmarker configuration.
        """

        def hash_smiles(smiles: list[str]) -> str:
            """
            Generate a hash for a list of SMILES strings.

            Args:
                smiles: List of SMILES strings.

            Returns:
                A unique hash string for the SMILES.
            """
            sorted_smiles = sorted(smiles)
            return hashlib.md5("\n".join(sorted_smiles).encode()).hexdigest()

        train_hash = hash_smiles(dataset.train_smiles)
        valid_hash = hash_smiles(dataset.validation_smiles)
        return f"{train_hash}_{valid_hash}_{num_samples_to_generate}"


def flatten_dict(d: dict, parent_key: str = "", sep: str = "/") -> dict:
    """
    Flatten a nested dictionary by joining keys with a separator.

    Args:
        d: Dictionary to flatten
        parent_key: Parent key for recursion
        sep: Separator to use between keys

    Returns:
        Flattened dictionary with keys joined by separator
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


# def get_smiles(samples: Iterable[MolecularSample]) -> list[str]:
#     smiles: list[str] = []
#     for sample in samples:
#         mol = make_molecule(sample["X"], sample["E"], sample["mask"])
#         if mol is not None:
#             smiles.append(Chem.MolToSmiles(mol))
#         else:
#             print(f"Invalid molecule in validation set: {sample}")
#     return smiles
