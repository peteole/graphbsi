from typing import Literal

import torch
from torch.nn.functional import one_hot
from tqdm import tqdm

from project.belief.categorical_belief import CategoricalBelief
from project.belief.composed_belief import ComposedBelief
from project.data.molecule import MoleculeDataModule
from project.data.sequence import SequenceDataModule
from project.data.synthetic_graphs import SyntheticGraphDataModule
from project.prior.softmax_normal_prior import SoftmaxNormalPrior
from project.scheduler.ole_scheduler import OLEScheduler
from project.scheduler.scheduler import Scheduler


def construct_molecular_belief_from_config(
    num_node_features: int,
    num_edge_features: int,
    max_num_atoms: int,
    datamodule: MoleculeDataModule,
    device: str = "cpu",
    scheduler: Scheduler = OLEScheduler(),
    initial_categorical_belief_std: float = 0.2,
    marginal_as_prior: bool = False,
    loss_mode: Literal["kl", "cross_entropy"] = "kl",
    **kwargs,
) -> ComposedBelief:
    node_marginals = None
    edge_marginals = None
    if marginal_as_prior:
        save_path = datamodule.cache_dir / f"{datamodule.fingerprint}.pt"
        if save_path.exists():
            print(f"Loading marginal probabilities from {save_path}")
            data = torch.load(save_path, weights_only=False)
            edge_marginals = data["edge_marginals"]
            node_marginals = data["node_marginals"]
        else:
            edge_class_counts = torch.zeros(num_edge_features)
            node_class_counts = torch.zeros(num_node_features)
            assert datamodule.train_dataset is not None, (
                "Train dataset must be provided for marginal prior."
            )
            for graph in tqdm(
                datamodule.train_dataset, desc="Calculating marginal probabilities"
            ):
                E = graph["E"]
                edge_class_counts += (
                    one_hot(E.long(), num_classes=num_edge_features)
                    .float()
                    .mean(dim=tuple(range(0, E.ndim)))
                    .cpu()
                )
                X = graph["X"]
                node_class_counts += (
                    one_hot(X.long(), num_classes=num_node_features)
                    .float()
                    .mean(dim=tuple(range(0, X.ndim)))
                    .cpu()
                )
            node_marginals = node_class_counts / node_class_counts.sum()
            edge_marginals = edge_class_counts / edge_class_counts.sum()
            save_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"Saving marginal probabilities to {save_path}")
            torch.save(
                {
                    "edge_marginals": edge_marginals,
                    "node_marginals": node_marginals,
                },
                save_path,
            )

    node_belief = CategoricalBelief(
        shape=(max_num_atoms,),
        num_classes=num_node_features,
        device=device,
        key="X",
        scheduler=scheduler,
        categorical_prior=SoftmaxNormalPrior(
            shape=(max_num_atoms,),
            num_classes=num_node_features,
            device=device,
            initial_belief_std=initial_categorical_belief_std,
            mean_probabilities=node_marginals,
        ),
        loss_mode=loss_mode,
    )
    edge_belief = CategoricalBelief(
        shape=(max_num_atoms, max_num_atoms),
        num_classes=num_edge_features,
        device=device,
        key="E",
        scheduler=scheduler,
        categorical_prior=SoftmaxNormalPrior(
            shape=(max_num_atoms, max_num_atoms),
            num_classes=num_edge_features,
            device=device,
            initial_belief_std=initial_categorical_belief_std,
            mean_probabilities=edge_marginals,
        ),
        loss_mode=loss_mode,
    )
    graph_belief = ComposedBelief([node_belief, edge_belief], device=device)
    return graph_belief


def construct_synthetic_graph_belief_from_config(
    max_num_nodes: int,
    datamodule: SyntheticGraphDataModule,
    device: str = "cpu",
    scheduler: Scheduler = OLEScheduler(),
    initial_categorical_belief_std: float = 0.2,
    marginal_as_prior: bool = False,
    loss_mode: Literal["kl", "cross_entropy"] = "kl",
    **kwargs,
) -> ComposedBelief:
    mean_probabilities = None
    if marginal_as_prior:
        edge_present_counts = 0
        edge_not_present_counts = 0
        assert datamodule.train_samples is not None, (
            "Train samples must be provided for marginal prior."
        )
        for graph in tqdm(
            datamodule.train_samples, desc="Calculating marginal probabilities"
        ):
            edge_present_counts += (graph["E"] == 1).sum().cpu().item()
            edge_not_present_counts += (graph["E"] == 0).sum().cpu().item()
        mean_probabilities = [
            edge_not_present_counts / (edge_present_counts + edge_not_present_counts),
            edge_present_counts / (edge_present_counts + edge_not_present_counts),
        ]
    node_belief = CategoricalBelief(
        shape=(max_num_nodes,),
        num_classes=2,  # present or absent
        device=device,
        key="X",
        scheduler=scheduler,
        categorical_prior=SoftmaxNormalPrior(
            shape=(max_num_nodes,),
            num_classes=2,  # present or absent
            device=device,
            initial_belief_std=initial_categorical_belief_std,
        ),
        loss_mode=loss_mode,
    )
    edge_belief = CategoricalBelief(
        shape=(max_num_nodes, max_num_nodes),
        num_classes=2,  # present or absent
        device=device,
        key="E",
        scheduler=scheduler,
        categorical_prior=SoftmaxNormalPrior(
            shape=(max_num_nodes, max_num_nodes),
            num_classes=2,  # present or absent
            device=device,
            initial_belief_std=initial_categorical_belief_std,
            mean_probabilities=mean_probabilities,
        ),
        loss_mode=loss_mode,
    )
    graph_belief = ComposedBelief([node_belief, edge_belief], device=device)
    return graph_belief


def construct_sequence_belief_from_config(
    datamodule: SequenceDataModule,
    device: str = "cpu",
    scheduler: Scheduler = OLEScheduler(),
    initial_categorical_belief_std: float = 1.0,
    alphabet_size: int = 4,
    sequence_length: int | None = None,
    marginal_as_prior: bool = False,
    loss_mode: Literal["kl", "cross_entropy"] = "kl",
    **kwargs,
) -> CategoricalBelief:
    if sequence_length is None:
        if datamodule.sequence_length == 0:
            datamodule.setup(stage="fit")
        sequence_length = datamodule.sequence_length

    mean_probabilities = None
    if marginal_as_prior:
        nucleotide_counts = torch.zeros(alphabet_size)
        assert datamodule.train_dataset is not None, (
            "Train dataset must be provided for marginal prior."
        )

        print("Calculating nucleotide marginal probabilities...")
        for sample in tqdm(datamodule.train_dataset, desc="Processing sequences"):
            sequence = sample["sequence"]
            nucleotide_counts += (
                one_hot(sequence.long(), num_classes=alphabet_size)
                .float()
                .sum(dim=0)
                .cpu()
            )

        mean_probabilities = nucleotide_counts / nucleotide_counts.sum()
        print(f"Nucleotide marginal probabilities: {mean_probabilities}")
    print("Sequence length:", sequence_length)
    sequence_belief = CategoricalBelief(
        shape=(sequence_length,),
        num_classes=alphabet_size,
        device=device,
        key="sequence",
        scheduler=scheduler,
        categorical_prior=SoftmaxNormalPrior(
            shape=(sequence_length,),
            num_classes=alphabet_size,
            device=device,
            initial_belief_std=initial_categorical_belief_std,
            mean_probabilities=mean_probabilities,
        ),
        loss_mode=loss_mode,
    )

    return sequence_belief
