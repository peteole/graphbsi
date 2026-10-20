from collections import Counter

import torch
from tensordict import TensorDict
from torch import Tensor
from tqdm import tqdm

from project.class_conditioner.class_conditioner import ClassConditioner
from project.data.synthetic_graphs import SyntheticGraphDataModule


class NodeMaskClassConditioner(ClassConditioner):
    @classmethod
    def compute_node_count_distribution(cls, datamodule: SyntheticGraphDataModule):
        datamodule.setup("fit")
        if datamodule.train_samples is None:
            raise ValueError("Train samples are not set")

        num_node_counts: Counter[int] = Counter()
        for sample in tqdm(
            datamodule.train_samples, desc="Measuring node count distribution"
        ):
            num_node_counts[sample["mask"].sum().item()] += 1
        # self.num_node_counts = num_node_counts
        max_node_count = datamodule.max_num_nodes
        num_samples_per_node_count = [
            num_node_counts[i] for i in range(max_node_count + 1)
        ]
        node_count_probs = torch.tensor(num_samples_per_node_count) / sum(
            num_samples_per_node_count
        )
        node_count_distribution = torch.distributions.Categorical(node_count_probs)
        return node_count_distribution, max_node_count

    @classmethod
    def from_config(cls, datamodule: SyntheticGraphDataModule, **kwargs):
        # if datamodule.dataset_folder is None:
        #     raise ValueError("Dataset folder is not set in the datamodule")
        # cache_path = datamodule.dataset_folder / "node_mask_class_conditioner.pt"
        # if cache_path.exists():
        #     print(f"Loading AtomMaskClassConditioner from {cache_path}")
        #     config = torch.load(cache_path, weights_only=False)
        #     return cls(
        #         node_count_distribution=config["node_count_distribution"],
        #         max_node_count=config["max_node_count"],
        #     )
        # print("Computing node count distribution...")
        node_count_distribution, max_node_count = cls.compute_node_count_distribution(
            datamodule
        )
        # print("Saving AtomMaskClassConditioner to cache...")
        # torch.save(
        #     {
        #         "node_count_distribution": node_count_distribution,
        #         "max_node_count": max_node_count,
        #     },
        #     cache_path,
        # )
        # print(f"Saved AtomMaskClassConditioner to {cache_path}")
        return cls(
            node_count_distribution=node_count_distribution,
            max_node_count=max_node_count,
        )

    def __init__(
        self,
        node_count_distribution: torch.distributions.Categorical,
        max_node_count: int,
    ):
        """
        Initialize the AtomMaskClassConditioner with a distribution of node counts and the maximum node count.
        """
        super().__init__()
        self.node_count_distribution = node_count_distribution
        self.max_node_count = max_node_count

    def classify_sample(self, sample: TensorDict) -> Tensor:
        return sample["mask"]

    def write_sample_classifications_to_belief(
        self, classifications: Tensor, belief: TensorDict
    ) -> None:
        belief["mask"] = classifications

    def sample_classifications(self, num_samples, device=None) -> Tensor:
        """
        Sample the node count from the distribution of node counts. Then, create a random mask for each count.
        """
        node_counts = self.node_count_distribution.sample((num_samples,)).to(device)
        node_masks = torch.zeros(
            num_samples, self.max_node_count, dtype=torch.bool, device=device
        )

        # first set node_masks[i,:node_counts[i]] = True
        I, J = torch.meshgrid(
            torch.arange(num_samples, device=device),
            torch.arange(self.max_node_count, device=device),
            indexing="ij",
        )
        node_masks[I, J] = J < node_counts[I]

        # then randomly permute the nodes in dim=-1
        # get a batch of random permutations
        random_matrix = torch.rand(
            num_samples, self.max_node_count, device=node_masks.device
        )
        batch_perm = random_matrix.argsort(dim=1)
        node_masks = node_masks.gather(dim=1, index=batch_perm)

        return node_masks
