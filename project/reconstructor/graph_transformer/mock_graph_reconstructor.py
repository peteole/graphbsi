import torch
from tensordict import TensorDict

from project.reconstructor.reconstructor import SampleReconstructor
from project.utils.tensor_utils import dict_of_tensors_to_tensor_dict


class MockMoleculeReconstructor(SampleReconstructor):
    def __init__(self, molecules: TensorDict | dict[str, torch.Tensor]):
        super().__init__()
        if not isinstance(molecules, TensorDict):
            molecules = dict_of_tensors_to_tensor_dict(molecules)
        # Datasets store atom/bond types in compact unsigned dtypes that indexing
        # ops do not support; widen them.
        molecules = molecules.apply(
            lambda t: t.long()
            if not t.is_floating_point() and t.dtype != torch.bool
            else t
        )
        masks = molecules["mask"]
        atom_counts = masks.sum(dim=1).tolist()
        self.molecule_by_atom_count: dict[int, TensorDict] = {}
        for i, atom_count in enumerate(atom_counts):
            if atom_count not in self.molecule_by_atom_count:
                self.molecule_by_atom_count[atom_count] = molecules[i]

    def forward(self, belief: TensorDict) -> TensorDict:
        atom_counts = belief["mask"].sum(dim=1).tolist()
        reconstructions = []
        for i, atom_count in enumerate(atom_counts):
            mask = belief["mask"][i]
            if atom_count in self.molecule_by_atom_count:
                # permute the molecule to match the mask
                molecule = self.molecule_by_atom_count[atom_count]
                X_permuted = torch.empty_like(molecule["X"])
                X_permuted[mask] = molecule["X"][:atom_count]
                X_permuted[~mask] = molecule["X"][atom_count:]

                # permute adjacency matrix - need to permute both dimensions
                E_original = molecule["E"]

                # Create inverse permutation: map original molecule indices to target indices
                # Original molecule has atoms at positions [0, 1, 2, ..., atom_count-1, atom_count, ...]
                # Target has active atoms where mask=True, inactive atoms where mask=False
                target_indices = torch.empty(E_original.shape[0], dtype=torch.long)
                target_indices[mask] = torch.arange(atom_count)
                target_indices[~mask] = torch.arange(atom_count, E_original.shape[0])

                # Apply inverse permutation to adjacency matrix
                E_permuted = E_original[target_indices][:, target_indices]

                # Create reconstruction for this sample
                reconstruction = TensorDict(
                    {"X": X_permuted, "E": E_permuted, "mask": mask}, batch_size=[]
                )
                reconstructions.append(reconstruction)

            else:
                raise ValueError(
                    f"No molecule found for atom count {atom_count} in mock atoms."
                )

        # Stack all reconstructions into a batch and return them as one-hot
        # class probabilities, like a real reconstructor.
        out = TensorDict.stack(reconstructions, dim=0)
        for key in ("X", "E"):
            out[key] = torch.nn.functional.one_hot(
                out[key], num_classes=belief[key].shape[-1]
            ).float()
        return out  # type: ignore[return-value]
