from typing import Literal, TypedDict

import lightning.pytorch as pl
import networkx as nx
import numpy as np
import torch
from synthetic_graph_benchmarks.dataset import Dataset
from torch.utils.data import DataLoader


class SyntheticGraphSample(TypedDict):
    X: torch.Tensor
    E: torch.Tensor
    mask: torch.Tensor


def sample_to_graph(sample: SyntheticGraphSample) -> nx.Graph:
    mask = sample["mask"].cpu().numpy()
    E = sample["E"].cpu().numpy()
    if E.ndim == 3:
        E = E.argmax(axis=-1)
    else:
        assert E.ndim == 2, "E should be either 2D or 3D"
        assert E.dtype in [
            np.int8,
            np.int16,
            np.int32,
            np.int64,
            np.uint8,
            np.uint16,
            np.uint32,
            np.uint64,
        ], "E should be an integer type"
    E = E[mask][:, mask]
    g = nx.from_numpy_array(E, create_using=nx.Graph)
    return g


def graph_to_sample(g: nx.Graph, max_num_nodes: int) -> SyntheticGraphSample:
    num_nodes = g.number_of_nodes()
    E = torch.zeros((max_num_nodes, max_num_nodes), dtype=torch.uint8)
    E[:num_nodes, :num_nodes] = torch.tensor(nx.to_numpy_array(g), dtype=torch.uint8)
    mask = torch.zeros((max_num_nodes,), dtype=torch.bool)
    mask[:num_nodes] = True
    X = mask.to(torch.uint8)
    return SyntheticGraphSample(X=X, E=E, mask=mask)


class SyntheticGraphDataModule(pl.LightningDataModule):
    @classmethod
    def from_config(
        cls,
        ds_type: Literal["planar", "sbm", "tree"],
        max_num_nodes: int,
        cache_dir: str,
        batch_size: int = 32,
        **kwargs,
    ):
        if ds_type == "planar":
            if max_num_nodes != 64:
                print(
                    "Warning: All graphs in the planar dataset have 64 nodes but set max_num_nodes to",
                    max_num_nodes,
                )
            dataset = Dataset.load_planar(cache_dir=cache_dir)
        elif ds_type == "sbm":
            dataset = Dataset.load_sbm(cache_dir=cache_dir)
        elif ds_type == "tree":
            if max_num_nodes != 64:
                print(
                    "Warning: All graphs in the tree dataset have 64 nodes but set max_num_nodes to",
                    max_num_nodes,
                )
            dataset = Dataset.load_tree(cache_dir=cache_dir)
        else:
            raise ValueError(f"Unknown dataset type: {ds_type}")
        return cls(dataset=dataset, batch_size=batch_size, max_num_nodes=max_num_nodes)

    def __init__(self, dataset: Dataset, max_num_nodes: int, batch_size: int = 32):
        super().__init__()
        self.max_num_nodes = max_num_nodes
        self.dataset = dataset
        self.batch_size = batch_size
        self.train_samples: list[SyntheticGraphSample] | None = None
        self.val_samples: list[SyntheticGraphSample] | None = None
        self.test_samples: list[SyntheticGraphSample] | None = None

    def prepare_data(self):
        # Load the dataset and split it into train, validation, and test sets
        # max_num_nodes = max([g.number_of_nodes() for g in self.dataset.train_graphs] + [g.number_of_nodes() for g in self.dataset.val_graphs] + [g.number_of_nodes() for g in self.dataset.test_graphs or []])

        def _convert_samples(graphs: list[nx.Graph]):
            samples = [graph_to_sample(g, self.max_num_nodes) for g in graphs]
            return samples

        self.train_samples = _convert_samples(self.dataset.train_graphs)
        self.val_samples = _convert_samples(self.dataset.val_graphs)
        self.test_samples = _convert_samples(self.dataset.test_graphs or [])

    def setup(self, stage: str | None = None):
        # This method is called to set up the data module
        pass

    def train_dataloader(self):
        assert self.train_samples is not None, (
            "Train samples are not prepared. Call prepare_data() first."
        )
        return DataLoader(
            self.train_samples, batch_size=self.batch_size, shuffle=True, pin_memory=True
        )

    def val_dataloader(self):
        assert self.val_samples is not None, (
            "Validation samples are not prepared. Call prepare_data() first."
        )
        return DataLoader(
            self.val_samples, batch_size=self.batch_size, shuffle=False, pin_memory=True
        )

    def test_dataloader(self):
        assert self.test_samples is not None, (
            "Test samples are not prepared. Call prepare_data() first."
        )
        return DataLoader(self.test_samples, batch_size=self.batch_size)

    def get_max_num_nodes(self) -> int:
        return self.max_num_nodes
