from collections import Counter

import torch
from tensordict import TensorDict
from torch import Tensor
from tqdm import tqdm

from project.class_conditioner.class_conditioner import ClassConditioner
from project.data.molecule import MoleculeDataModule


class AtomMaskClassConditioner(ClassConditioner):
    @classmethod
    def compute_atom_count_distribution(cls, datamodule: MoleculeDataModule):
        datamodule.setup("fit")
        if datamodule.train_dataset is None:
            raise ValueError("Train dataset is not set")

        num_atom_counts: Counter[int] = Counter()
        for sample in tqdm(
            datamodule.train_dataset, desc="Measuring atom count distribution"
        ):
            num_atom_counts[sample["mask"].sum().item()] += 1
        # self.num_atom_counts = num_atom_counts
        max_atom_count = datamodule.get_max_num_atoms()
        num_samples_per_atom_count = [
            num_atom_counts[i] for i in range(max_atom_count + 1)
        ]
        atom_count_probs = torch.tensor(num_samples_per_atom_count) / sum(
            num_samples_per_atom_count
        )
        atom_count_distribution = torch.distributions.Categorical(atom_count_probs)
        return atom_count_distribution, max_atom_count

    @classmethod
    def from_config(cls, datamodule: MoleculeDataModule, **kwargs):
        if datamodule.dataset_folder is None:
            raise ValueError("Dataset folder is not set in the datamodule")
        cache_path = datamodule.dataset_folder / "atom_mask_class_conditioner.pt"
        if cache_path.exists():
            print(f"Loading AtomMaskClassConditioner from {cache_path}")
            config = torch.load(cache_path, weights_only=False)
            return cls(
                atom_count_distribution=config["atom_count_distribution"],
                max_atom_count=config["max_atom_count"],
            )
        print("Computing atom count distribution...")
        atom_count_distribution, max_atom_count = cls.compute_atom_count_distribution(
            datamodule
        )
        print("Saving AtomMaskClassConditioner to cache...")
        torch.save(
            {
                "atom_count_distribution": atom_count_distribution,
                "max_atom_count": max_atom_count,
            },
            cache_path,
        )
        print(f"Saved AtomMaskClassConditioner to {cache_path}")
        return cls(
            atom_count_distribution=atom_count_distribution,
            max_atom_count=max_atom_count,
        )

    def __init__(
        self,
        atom_count_distribution: torch.distributions.Categorical,
        max_atom_count: int,
    ):
        """
        Initialize the AtomMaskClassConditioner with a distribution of atom counts and the maximum atom count.
        """
        super().__init__()
        self.atom_count_distribution = atom_count_distribution
        self.max_atom_count = max_atom_count

    def classify_sample(self, sample: TensorDict) -> Tensor:
        return sample["mask"]

    def write_sample_classifications_to_belief(
        self, classifications: Tensor, belief: TensorDict
    ) -> None:
        belief["mask"] = classifications

    def sample_classifications(self, num_samples, device=None) -> Tensor:
        """
        Sample the atom count from the distribution of atom counts. Then, create a random mask for each count.
        """
        atom_counts = self.atom_count_distribution.sample((num_samples,)).to(device)
        atom_masks = torch.zeros(
            num_samples, self.max_atom_count, dtype=torch.bool, device=device
        )

        # first set atom_masks[i,:atom_counts[i]] = True
        I, J = torch.meshgrid(
            torch.arange(num_samples, device=device),
            torch.arange(self.max_atom_count, device=device),
            indexing="ij",
        )
        atom_masks[I, J] = J < atom_counts[I]

        # then randomly permute the atoms in dim=-1
        # get a batch of random permutations
        random_matrix = torch.rand(
            num_samples, self.max_atom_count, device=atom_masks.device
        )
        batch_perm = random_matrix.argsort(dim=1)
        atom_masks = atom_masks.gather(dim=1, index=batch_perm)

        return atom_masks
