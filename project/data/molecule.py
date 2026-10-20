import json
import math
import os
import shutil
from abc import abstractmethod
from functools import partial
from pathlib import Path
from typing import (
    Any,
    Callable,
    Counter,
    Iterable,
    Literal,
    Optional,
    Protocol,
    Set,
    TypedDict,
)

import h5py
import lightning.pytorch as pl
import numpy as np
import torch
from molecule_benchmarks import SmilesDataset
from molecule_benchmarks.utils import available_cpu_count, mapper
from rdkit import Chem
from rdkit.RDLogger import CRITICAL, logger
from torch.utils.data import DataLoader
from tqdm import tqdm

from project.utils.tensor_utils import SparseTensorH5Dataset

lg = logger()
lg.setLevel(CRITICAL)


def _molecule_n_jobs() -> int:
    """Worker count for SMILES preprocessing. Default 1 (ProcessPool is flaky on NFS)."""
    raw = os.environ.get("GRAPHBSI_MOLECULE_N_JOBS", "1")
    try:
        return max(1, int(raw))
    except ValueError:
        return 1


def _map_smiles(
    func: Callable[..., Any],
    iterable: Iterable[Any],
    *,
    job_name: str,
) -> list[Any]:
    n_jobs = _molecule_n_jobs()
    try:
        return mapper(n_jobs=n_jobs, job_name=job_name)(func, iterable)
    except Exception as exc:
        if n_jobs == 1:
            raise
        print(
            f"{job_name}: parallel preprocessing failed ({type(exc).__name__}: {exc}); "
            "retrying with n_jobs=1"
        )
        return mapper(n_jobs=1, job_name=job_name)(func, iterable)


class MolecularSample(TypedDict):
    X: torch.Tensor
    E: torch.Tensor
    mask: torch.Tensor


class MoleculeSupplier(Protocol):
    """Protocol for a supplier that provides molecules."""

    @abstractmethod
    def get_train_molecules(
        self, num_samples: Optional[int] = None
    ) -> Iterable[Chem.Mol | None]:
        """Get an iterable of molecules, optionally limited to num_samples."""
        pass

    @abstractmethod
    def get_val_molecules(
        self, num_samples: Optional[int] = None
    ) -> Iterable[Chem.Mol | None]:
        """Get an iterable of validation molecules, optionally limited to num_samples."""
        pass


class SmilesMoleculeSupplier(MoleculeSupplier):
    def __init__(self, smiles_dataset: SmilesDataset):
        self.smiles_dataset = smiles_dataset

    def get_train_molecules(
        self, num_samples: Optional[int] = None
    ) -> Iterable[Chem.Mol | None]:
        train_smiles = self.smiles_dataset.get_train_smiles()
        if num_samples is not None:
            train_smiles = train_smiles[:num_samples]
        return (Chem.MolFromSmiles(smiles) for smiles in train_smiles)

    def get_val_molecules(
        self, num_samples: Optional[int] = None
    ) -> Iterable[Chem.Mol | None]:
        val_smiles = self.smiles_dataset.get_validation_smiles()
        if num_samples is not None:
            val_smiles = val_smiles[:num_samples]
        return (Chem.MolFromSmiles(smiles) for smiles in val_smiles)


class MoleculeDataModule(pl.LightningDataModule):
    @classmethod
    def from_config(
        cls,
        ds_type: Literal["qm9", "guacamol", "moses"],
        cache_dir: str | Path,
        max_num_atoms: int,
        num_atom_types: int,
        num_bond_types: int,
        batch_size: int = 32,
        remove_hydrogen: bool = True,
        remove_charges: bool = False,
        max_num_samples: int | None = None,
        random_seed: int = 42,
        num_workers: int = 0,
        **kwargs,
    ) -> "MoleculeDataModule":
        cache_dir = Path(cache_dir)
        smiles_dataset = None
        if ds_type == "qm9":
            smiles_dataset = SmilesDataset.load_qm9_dataset(
                max_train_samples=max_num_samples, cache_dir=cache_dir / ds_type
            )
        elif ds_type == "guacamol":
            smiles_dataset = SmilesDataset.load_guacamol_dataset(
                max_train_samples=max_num_samples, cache_dir=cache_dir / ds_type
            )
        elif ds_type == "moses":
            smiles_dataset = SmilesDataset.load_moses_dataset(
                max_train_samples=max_num_samples, cache_dir=cache_dir / ds_type
            )
        else:
            raise ValueError(f"Unknown dataset type: {ds_type}")

        return cls(
            smiles_dataset=smiles_dataset,
            cache_dir=cache_dir / ds_type,
            max_num_atoms=max_num_atoms,
            num_atom_types=num_atom_types,
            num_bond_types=num_bond_types,
            batch_size=batch_size,
            remove_hydrogen=remove_hydrogen,
            remove_charges=remove_charges,
            max_num_samples=max_num_samples,
            random_seed=random_seed,
        )

    def __init__(
        self,
        smiles_dataset: SmilesDataset,
        cache_dir: str | Path,
        max_num_atoms: int,
        num_atom_types: int,
        num_bond_types: int,
        batch_size: int = 32,
        remove_hydrogen: bool = True,
        remove_charges: bool = False,
        max_num_samples: int | None = None,
        random_seed: int = 42,
    ):
        super().__init__()
        self.smiles_dataset = smiles_dataset
        self.batch_size = batch_size
        self.num_atom_types = num_atom_types
        self.num_bond_types = num_bond_types
        self.cache_dir = Path(cache_dir)
        self.remove_hydrogen = remove_hydrogen
        self.remove_charges = remove_charges
        self.max_num_atoms = max_num_atoms
        self.dataset_folder: Path | None = None
        self.train_length: int = 0
        self.val_length: int = 0

        self.atom_types: dict[str, int] | None = None
        self.bond_types: dict[Chem.BondType | None, int] | None = None
        self.max_num_samples = max_num_samples
        self.random_seed = random_seed
        self.train_dataset: Optional[SparseTensorH5Dataset] = None
        self.val_dataset: Optional[SparseTensorH5Dataset] = None
        self.test_dataset: Optional[SparseTensorH5Dataset] = None

    @property
    def fingerprint(self) -> str:
        """Return a fingerprint string that uniquely identifies the dataset configuration."""
        return f"molecule_data_num_samples={self.max_num_samples}_num_atoms={self.max_num_atoms}_num_atom_types={self.num_atom_types}_num_bond_types={self.num_bond_types}_remove_hydrogen={self.remove_hydrogen}_remove_charges={self.remove_charges}"
    def prepare_data(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Create dataset folder path
        self.dataset_folder = (
            self.cache_dir
            / self.fingerprint
        )
        metadata_path = self.dataset_folder / "metadata.json"

        if metadata_path.exists():
            print(f"Loading cached data from {self.dataset_folder}")
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            # Load atom types
            self.atom_types = metadata["atom_types"]

            # Load bond types (convert bond type names back to enums)
            self.bond_types = {}
            bond_type_map = {
                "None": None,
                "SINGLE": Chem.BondType.SINGLE,
                "DOUBLE": Chem.BondType.DOUBLE,
                "TRIPLE": Chem.BondType.TRIPLE,
                "AROMATIC": Chem.BondType.AROMATIC,
            }
            for bond_name, idx in metadata["bond_types"].items():
                self.bond_types[bond_type_map[bond_name]] = idx

            # Load dataset lengths
            self.train_length = metadata["train_length"]
            self.val_length = metadata["val_length"]

            if self.atom_types is None or self.bond_types is None:
                raise ValueError("Atom or bond types are not loaded from cache.")
            if self.train_length == 0 or self.val_length == 0:
                raise ValueError("Cached dataset lengths are zero.")
            return

        # Create dataset folder
        # clear any existing files
        shutil.rmtree(self.dataset_folder, ignore_errors=True)
        self.dataset_folder.mkdir(parents=True, exist_ok=True)

        # Process samples and save to HDF5
        samples_raw = (
            self.smiles_dataset.get_train_smiles()
            + self.smiles_dataset.get_validation_smiles()
        )
        existing_atoms: Counter[str] = Counter()

        # Parallelize atom type gathering
        print("Gathering present atom types...")
        # for smiles in tqdm(samples_raw, desc="Processing SMILES for atom types"):
        #     atom_counter = MoleculeDataModule._process_smiles_for_atoms(
        #         smiles,
        #         remove_hydrogen=self.remove_hydrogen,
        #         remove_charges=self.remove_charges,
        #     )
        #     existing_atoms.update(atom_counter)
        atom_counters = _map_smiles(
            partial(
                MoleculeDataModule._process_smiles_for_atoms,
                remove_hydrogen=self.remove_hydrogen,
                remove_charges=self.remove_charges,
            ),
            samples_raw,
            job_name="Counting atoms",
        )
        # Combine all counters
        max_total_atom_count = 0
        max_heavy_atom_count = 0  # Maximum number of non-hydrogen atoms in a molecule
        for counter in atom_counters:
            existing_atoms.update(counter)
            max_total_atom_count = max(max_total_atom_count, sum(counter.values()))
            max_heavy_atom_count = max(
                max_heavy_atom_count,
                sum(v for k, v in counter.items() if k != "H" and not k.startswith("H,")),
            )
        print("Maximum total atom count in a molecule:", max_total_atom_count)
        print("Maximum heavy atom count in a molecule:", max_heavy_atom_count)
        assert len(existing_atoms.keys()) <= self.num_atom_types, (
            f"Number of unique atom types {len(existing_atoms.keys())} exceeds the limit {self.num_atom_types}."
        )

        atom_types = {
            symbol: idx + 1
            for idx, (symbol, count) in enumerate(
                sorted(existing_atoms.items(), key=lambda x: x[1], reverse=True)
            )
        }
        atom_types["None"] = 0  # Add a not present type
        print("Atom types:", atom_types)

        if len(atom_types) != self.num_atom_types:
            print(
                f"Warning: Expected {self.num_atom_types} atom types, but got {len(atom_types)}."
            )

        bond_types = {
            bond_type: idx
            for idx, bond_type in enumerate(
                [
                    None,
                    Chem.BondType.SINGLE,
                    Chem.BondType.DOUBLE,
                    Chem.BondType.TRIPLE,
                    Chem.BondType.AROMATIC,
                ][: self.num_bond_types]
            )
        }
        print("Bond types:", bond_types)

        self.atom_types = atom_types
        self.bond_types = bond_types

        # Process and save data to HDF5
        train_smiles = self.smiles_dataset.get_train_smiles()
        val_smiles = self.smiles_dataset.get_validation_smiles()

        self.train_length = self._process_and_save_to_h5_batched(
            train_smiles, self.dataset_folder / "train_graphs.h5"
        )
        self.val_length = self._process_and_save_to_h5_batched(
            val_smiles, self.dataset_folder / "val_graphs.h5"
        )

        # Save metadata to JSON
        metadata = {
            "atom_types": self.atom_types,
            "bond_types": {
                ("None" if bond_type is None else str(bond_type).split(".")[-1]): idx
                for bond_type, idx in self.bond_types.items()
            },
            "train_length": self.train_length,
            "val_length": self.val_length,
            "max_num_atoms": self.max_num_atoms,
            "num_atom_types": self.num_atom_types,
            "num_bond_types": self.num_bond_types,
            "remove_hydrogen": self.remove_hydrogen,
            "remove_charges": self.remove_charges,
        }

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    def _process_and_save_to_h5_batched(
        self, smiles: Iterable[str], h5_file_path: Path
    ) -> int:
        """Process SMILES in batches and save to HDF5 file. Returns the number of processed samples."""
        # Convert to list if needed for parallelization
        smiles_list = list(smiles)
        total_samples = len(smiles_list)
        batch_size = 25000  # Process in batches of 25k

        # Ensure atom_types and bond_types are initialized
        if self.atom_types is None or self.bond_types is None:
            raise ValueError(
                "Atom types and bond types must be initialized before processing"
            )

        print(f"Processing {total_samples} SMILES strings in batches of {batch_size}...")

        # Initialize HDF5 file
        with h5py.File(h5_file_path, "w") as f:
            # Create datasets with maximum size
            x_dataset = f.create_dataset(
                "X",
                (total_samples, self.max_num_atoms),
                dtype=np.uint16,
                chunks=(1, self.max_num_atoms),
            )
            e_dataset = f.create_dataset(
                "E",
                (total_samples, self.max_num_atoms, self.max_num_atoms),
                dtype=np.uint8,
                chunks=(1, self.max_num_atoms, self.max_num_atoms),
            )
            mask_dataset = f.create_dataset(
                "mask",
                (total_samples, self.max_num_atoms),
                dtype=np.bool_,
                chunks=(1, self.max_num_atoms),
            )

            valid_count = 0

            # Process in batches
            num_batches = math.ceil(total_samples / batch_size)

            for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, total_samples)
                batch_smiles = smiles_list[start_idx:end_idx]

                # Process batch (default single-process; see GRAPHBSI_MOLECULE_N_JOBS)
                processed_results = _map_smiles(
                    partial(
                        self._process_single_smiles,
                        atom_types=self.atom_types,
                        bond_types=self.bond_types,
                        max_num_atoms=self.max_num_atoms,
                        remove_hydrogen=self.remove_hydrogen,
                        remove_charges=self.remove_charges,
                    ),
                    batch_smiles,
                    job_name="Converting SMILES to graphs",
                )
                # processed_results = map(
                #     partial(
                #         self._process_single_smiles,
                #         atom_types=self.atom_types,
                #         bond_types=self.bond_types,
                #         max_num_atoms=self.max_num_atoms,
                #         remove_hydrogen=self.remove_hydrogen,
                #         remove_charges=self.remove_charges,
                #     ),
                #     batch_smiles,
                # )
                print("Done processing batch of SMILES")
                # Filter out None results (failed processing)
                if any(result is None for result in processed_results):
                    print(
                        f"Warning: Some molecules in batch {batch_idx} could not be processed:"
                    )
                    # Optionally, you could log the specific SMILES strings that failed
                    for result, smiles in zip(processed_results, batch_smiles):
                        if result is None:
                            print(f"Failed to process SMILES: {smiles}")

                valid_samples = [
                    result for result in processed_results if result is not None
                ]

                if valid_samples:
                    # Save batch to HDF5
                    batch_size_actual = len(valid_samples)
                    for i, sample in tqdm(
                        enumerate(valid_samples),
                        desc="Saving batch to HDF5",
                        total=len(valid_samples),
                    ):
                        x_dataset[valid_count + i] = sample["X"].numpy()
                        e_dataset[valid_count + i] = sample["E"].numpy()
                        mask_dataset[valid_count + i] = sample["mask"].numpy()

                    valid_count += batch_size_actual

                # Free memory
                del processed_results
                del valid_samples

            # Resize datasets to actual size
            if valid_count < total_samples:
                print(
                    f"Warning: Expected {total_samples} valid samples, but got {valid_count}."
                )
                x_dataset.resize((valid_count, self.max_num_atoms))
                e_dataset.resize((valid_count, self.max_num_atoms, self.max_num_atoms))
                masks = mask_dataset[:valid_count]
                del f["mask"]
                f.create_dataset(
                    "mask",
                    data=masks,
                    dtype="bool",
                )

        print(f"Successfully processed {valid_count} out of {total_samples} molecules")
        return valid_count

    def setup(self, stage: str) -> None:
        if self.dataset_folder is None:
            raise ValueError("Dataset folder not set. Call prepare_data() first.")

        # num_classes = {
        #     "X": self.num_atom_types,
        #     "E": self.num_bond_types,
        # }
        num_classes = {} # one hot encoding in dataloader no longer needed
        # Create datasets that read from HDF5
        # Each dataset will open its own file handle
        self.train_dataset = SparseTensorH5Dataset(
            h5_file_path=self.dataset_folder / "train_graphs.h5",
            group_name="",  # No group name since we're using the root
            num_classes=num_classes,
            length=self.train_length,
        )
        self.val_dataset = SparseTensorH5Dataset(
            h5_file_path=self.dataset_folder / "val_graphs.h5",
            group_name="",  # No group name since we're using the root
            num_classes=num_classes,
            length=self.val_length,
        )
        self.test_dataset = SparseTensorH5Dataset(
            h5_file_path=self.dataset_folder
            / "val_graphs.h5",  # Use validation samples as test samples
            group_name="",  # No group name since we're using the root
            num_classes=num_classes,
            length=self.val_length,
        )

    def process_molecule(
        self,
        mol: Chem.Mol,
    ):
        Chem.Kekulize(mol)
        assert self.atom_types is not None, (
            "Atom types must be initialized before processing molecules."
        )
        assert self.bond_types is not None, (
            "Bond types must be initialized before processing molecules."
        )
        x = torch.zeros((self.max_num_atoms,), dtype=torch.long)
        e = torch.zeros((self.max_num_atoms, self.max_num_atoms), dtype=torch.long)
        mask = torch.zeros((self.max_num_atoms,), dtype=torch.bool)
        atom_idx = 0
        mol_to_dense_index_map: dict[int, int] = {}

        hydrogen_indices: Set[int] = set()
        for i, atom in enumerate(mol.GetAtoms()):
            symbol = atom.GetSymbol()
            if self.remove_hydrogen and symbol == "H":
                hydrogen_indices.add(atom.GetIdx())
                continue
            assert atom_idx < self.max_num_atoms, (
                f"May only have up to {self.max_num_atoms} atoms, but got {atom_idx + 1}."
            )
            if not self.remove_charges:
                symbol = f"{symbol},{atom.GetFormalCharge()}"
            if symbol not in self.atom_types:
                continue
            x[atom_idx] = self.atom_types[symbol]
            mask[atom_idx] = True
            mol_to_dense_index_map[i] = atom_idx
            atom_idx += 1
        for bond in mol.GetBonds():
            # Skip if start or end atom is H
            if (
                bond.GetBeginAtomIdx() in hydrogen_indices
                or bond.GetEndAtomIdx() in hydrogen_indices
            ):
                continue
            begin_idx = mol_to_dense_index_map[bond.GetBeginAtomIdx()]
            end_idx = mol_to_dense_index_map[bond.GetEndAtomIdx()]
            bond_type = bond.GetBondType()
            e[begin_idx, end_idx] = self.bond_types[bond_type]
            e[end_idx, begin_idx] = self.bond_types[bond_type]

        sample = MolecularSample(X=x, E=e, mask=mask)
        try:
            restored = self.molecular_sample_to_molecule(sample)
            if restored is not None:
                Chem.SanitizeMol(restored)
        except Exception as e:
            raise ValueError(
                f"Failed to restore molecule from sample: {e}. Sample: {sample}"
            ) from e
        if restored is not None:
            restored_smiles = get_unique_smiles(restored)
            original_smiles = get_unique_smiles(mol)
            if restored_smiles != original_smiles:
                print(
                    f"Warning: Restored SMILES {restored_smiles} does not match original SMILES {original_smiles}."
                )
        return sample

    def molecular_sample_to_molecule(
        self,
        sample: MolecularSample,
    ) -> Chem.Mol | None:
        """Convert a molecular sample to an RDKit molecule, or return None if invalid."""
        assert self.atom_types is not None, (
            "Atom types must be initialized before converting samples to molecules."
        )
        assert self.bond_types is not None, (
            "Bond types must be initialized before converting samples to molecules."
        )
        mol = Chem.RWMol()
        atom_indices = []
        x = sample["X"]
        if x.ndim == 2:
            x = x.argmax(dim=-1)  # Handle one-hot encoded atom types
        e = sample["E"]
        if e.ndim == 3:
            e = e.argmax(dim=-1)
        mask = sample["mask"]
        active_indices = mask.nonzero(as_tuple=True)[0]
        index_to_atom = {v: k for k, v in self.atom_types.items()}
        for i in active_indices:
            atom_type_index = int(x[i].item())
            if atom_type_index not in index_to_atom:
                return None
            atom_symbol = index_to_atom[atom_type_index]
            if atom_symbol == "None":
                return None
            if "," in atom_symbol:
                symbol, charge = atom_symbol.split(",")
                atom = Chem.Atom(symbol)
                atom.SetFormalCharge(int(charge))
            else:
                atom = Chem.Atom(atom_symbol)
            atom_idx = mol.AddAtom(atom)
            atom_indices.append(atom_idx)
        index_to_bond = {v: k for k, v in self.bond_types.items()}
        for i, i_atom_idx in zip(active_indices, atom_indices):
            for j, j_atom_idx in zip(active_indices, atom_indices):
                if i >= j or e[i, j] == 0:
                    continue
                bond_type_index = int(e[i, j].item())
                bond_type = index_to_bond[bond_type_index]
                if bond_type is None:
                    continue
                mol.AddBond(i_atom_idx, j_atom_idx, bond_type)
        mol = mol.GetMol()
        return mol

    def train_dataloader(self):
        assert self.train_dataset is not None, "Train dataset not initialized"
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=min(4, available_cpu_count()),
            persistent_workers=True,
            pin_memory=True,
            prefetch_factor=2,
        )

    def val_dataloader(self):
        assert self.val_dataset is not None, "Validation dataset not initialized"
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )

    def test_dataloader(self):
        assert self.test_dataset is not None, "Test dataset not initialized"
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )

    def get_num_node_features(self) -> int:
        """Get the number of node features."""
        return self.num_atom_types

    def get_num_edge_features(self) -> int:
        """Get the number of edge features."""
        return self.num_bond_types

    def get_max_num_atoms(self):
        return self.max_num_atoms

    def get_atom_type_names(self):
        assert self.atom_types is not None, "Atom types not initialized"
        inverted_atom_types = {v: k for k, v in self.atom_types.items()}
        return inverted_atom_types

    def get_bond_type_names(self):
        assert self.bond_types is not None, "Bond types not initialized"
        inverted_bond_types = {v: k for k, v in self.bond_types.items()}
        return inverted_bond_types

    @staticmethod
    def _process_smiles_for_atoms(
        smiles: str | None, remove_hydrogen: bool, remove_charges: bool
    ) -> Counter[str]:
        """Process a single SMILES string to extract atom types."""
        existing_atoms: Counter[str] = Counter()
        if smiles is None:
            return existing_atoms
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return existing_atoms
        if not remove_hydrogen:
            mol = Chem.AddHs(mol)

        for atom in mol.GetAtoms():
            symbol = atom.GetSymbol()
            if remove_hydrogen and symbol == "H":
                continue
            if not remove_charges:
                symbol = f"{symbol},{atom.GetFormalCharge()}"
            existing_atoms[symbol] += 1

        return existing_atoms

    @staticmethod
    def _process_single_smiles(
        smiles_str: str,
        atom_types: dict[str, int],
        bond_types: dict[Chem.BondType | None, int],
        max_num_atoms: int,
        remove_hydrogen: bool,
        remove_charges: bool,
    ) -> MolecularSample | None:
        """Process a single SMILES string into a molecular sample."""
        try:
            if smiles_str is None or smiles_str.strip() == "":
                return None
            mol = Chem.MolFromSmiles(smiles_str)
            if mol is None:
                return None
            return MoleculeDataModule._process_molecule_static(
                mol,
                atom_types,
                bond_types,
                max_num_atoms,
                remove_hydrogen,
                remove_charges,
            )
        except Exception as e:
            print(f"Warning: Failed to process molecule {smiles_str}: {e}")
            return None

    @staticmethod
    def _process_molecule_static(
        mol: Chem.Mol,
        atom_types: dict[str, int],
        bond_types: dict[Chem.BondType | None, int],
        max_num_atoms: int,
        remove_hydrogen: bool,
        remove_charges: bool,
    ) -> MolecularSample:
        """Static version of process_molecule for parallel processing."""
        Chem.Kekulize(mol)

        x = torch.zeros((max_num_atoms,), dtype=torch.uint16)
        e = torch.zeros((max_num_atoms, max_num_atoms), dtype=torch.uint8)
        mask = torch.zeros((max_num_atoms,), dtype=torch.bool)
        atom_idx = 0
        mol_to_dense_index_map: dict[int, int] = {}

        hydrogen_indices: Set[int] = set()
        for i, atom in enumerate(mol.GetAtoms()):
            symbol = atom.GetSymbol()
            if remove_hydrogen and symbol == "H":
                hydrogen_indices.add(atom.GetIdx())
                continue
            assert atom_idx < max_num_atoms, (
                f"May only have up to {max_num_atoms} atoms, but got {atom_idx + 1}."
            )
            if not remove_charges:
                symbol = f"{symbol},{atom.GetFormalCharge()}"
            if symbol not in atom_types:
                continue
            x[atom_idx] = atom_types[symbol]
            mask[atom_idx] = True
            mol_to_dense_index_map[i] = atom_idx
            atom_idx += 1

        for bond in mol.GetBonds():
            # Skip if start or end atom is H
            if (
                bond.GetBeginAtomIdx() in hydrogen_indices
                or bond.GetEndAtomIdx() in hydrogen_indices
            ):
                continue
            begin_idx = mol_to_dense_index_map[bond.GetBeginAtomIdx()]
            end_idx = mol_to_dense_index_map[bond.GetEndAtomIdx()]
            bond_type = bond.GetBondType()
            e[begin_idx, end_idx] = bond_types[bond_type]
            e[end_idx, begin_idx] = bond_types[bond_type]

        return MolecularSample(X=x, E=e, mask=mask)


def get_unique_smiles(mol: Chem.Mol) -> str:
    """Get a unique SMILES representation of the molecule."""
    # Chem.Kekulize(mol, clearAromaticFlags=True)
    Chem.SanitizeMol(mol)
    Chem.RemoveStereochemistry(mol)
    mol = Chem.AddHs(mol)
    mol = Chem.RemoveAllHs(mol)
    return Chem.MolToSmiles(mol)
