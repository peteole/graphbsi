from math import prod
from pathlib import Path
from typing import Optional

import h5py
import torch

# import nvtx
from tensordict import TensorDict


def expand_tensor(A: int | float | torch.Tensor, target_shape: tuple):
    """
    Expands a tensor A of shape (*batch,) to the target shape (*batch, *extra_shape) without making a copy.
    The batch size can have multiple dimensions.

    Args:
    - A (Tensor): Tensor of shape (*batch,)
    - target_shape (tuple): The target shape to expand to, including the batch dimension: target_shape = (*batch, *extra_shape)

    Returns:
    - Tensor: The expanded tensor with the target shape.
    """
    target_shape = torch.Size(target_shape)
    # Ensure the batch size in A matches the batch size in target_shape
    if not isinstance(A, torch.Tensor):
        A = torch.tensor(A)
    batch_size_A = A.shape
    batch_size_target = target_shape[: len(batch_size_A)]

    if batch_size_A != batch_size_target:
        raise ValueError(
            f"Batch size of A {batch_size_A} must match the batch size in target_shape {batch_size_target}."
        )

    # Reshape A to match the batch dimensions in the target shape
    expanded_A = A.view(
        *batch_size_target, *[1] * len(target_shape[len(batch_size_target) :])
    )

    # Use expand to broadcast to the full target shape without copying
    return expanded_A.expand(*target_shape)


def dict_of_tensors_to_tensor_dict(data: dict[str, torch.Tensor]) -> TensorDict:
    """
    Convert a dictionary of tensors to a TensorDict.
    Args:
        data (dict[str, torch.Tensor]): A dictionary where keys are strings and values are tensors.
    Returns:
        TensorDict: A TensorDict containing the same data.
    """
    return TensorDict(data, batch_size=[data[next(iter(data))].shape[0]])

def squared_l2_norm_tensordict(td: TensorDict) -> torch.Tensor:
    """
    Compute the squared L2 norm of each tensor in a TensorDict.

    Args:
        td (TensorDict): The TensorDict containing the tensors.

    Returns:
        torch.Tensor: The sum of the squared L2 norms of each tensor in the TensorDict, reduced for all dimensions except the batch size of the TensorDict.
    """
    l2 = 0
    for k,v in td.items():
        l2 += torch.norm(v, p=2, dim=tuple(range(1, v.ndim)), keepdim=True) ** 2
    return l2 # type: ignore

def flatten_tensordict(
    td: TensorDict,
    ignore_keys: set[str] = set(),
) -> torch.Tensor:
    """
    Flatten a TensorDict to a single tensor.
    Args:
        td (TensorDict): The TensorDict to flatten.
    Returns:
        torch.Tensor: A flattened tensor containing all values from the TensorDict. The keys are ordered alphabetically. Shape is (batch_size, sum of all flattened tensor sizes).
    """
    num_batchdims = len(td.batch_size)
    flattened_tensors = [
        td[key].flatten(start_dim=num_batchdims)
        if td[key].ndim > num_batchdims
        else td[key].unsqueeze(-1)
        for key in sorted(td.keys())
        if key not in ignore_keys
    ]
    concat = torch.cat(flattened_tensors, dim=1)
    assert concat.shape[:num_batchdims] == td.batch_size, (
        f"Batch size of concatenated tensor {concat.shape[:num_batchdims]} does not match TensorDict batch size {td.batch_size}."
    )
    return concat


def unflatten_tensor(
    tensor: torch.Tensor, shapes: dict[str, tuple], num_batchdims: int = 1
) -> TensorDict:
    """
    Unflatten a tensor to a TensorDict with specified batch size and shapes.

    Args:
        tensor (torch.Tensor): The flattened tensor to unflatten.
        batch_size (tuple): The batch size of the original TensorDict.
        shapes (dict[str, tuple]): A dictionary mapping keys to their original shapes (excluding batch size).

    Returns:
        TensorDict: The unflattened TensorDict.
    """
    tensors = {}
    start = 0
    batch_size = tensor.shape[:num_batchdims]
    for key, shape in sorted(shapes.items(), key=lambda x: x[0]):
        size = prod(shape)
        tensors[key] = tensor[:, start : start + size].view(
            *batch_size, *shape
        )  # Reshape to include batch size
        start += size
    assert start == tensor.shape[1], (
        f"Unflattened tensor size {start} does not match original tensor size {tensor.shape[1]}."
    )
    return TensorDict(tensors, batch_size=batch_size, device=tensor.device)


class SparseTensorDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset that saves tensors in a sparse format and one hot encodes them on-the-fly. If a key is not in num_classes, it is assumed to be a continuous tensor and is returned as is.
    """

    def __init__(
        self, samples: list[dict[str, torch.Tensor]], num_classes: dict[str, int]
    ):
        self.samples = samples
        self.num_classes = num_classes

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        one_hot_encoded_sample = {
            key: torch.nn.functional.one_hot(
                sample[key].long(), num_classes=self.num_classes[key]
            ).float()
            if key in self.num_classes
            else sample[key]
            for key in sample.keys()
        }
        return one_hot_encoded_sample


class SparseTensorH5Dataset(torch.utils.data.Dataset):
    """A dataset that reads sparse tensor data from HDF5 files on-demand."""

    def __init__(
        self,
        h5_file_path: Path,
        group_name: str,
        num_classes: dict[str, int],
        length: int,
        max_in_memory_size_bytes: int = 100 * 1024 * 1024,  # Default to 100 MB
    ):
        self.h5_file_path = h5_file_path
        self.group_name = group_name
        self.num_classes = num_classes
        self.length = length
        self._ds_file: Optional[h5py.File] = None
        self.cache: dict[str, torch.Tensor] = {}
        self.max_in_memory_size_bytes = max_in_memory_size_bytes
        for key in self.group.keys():
            dataset = self.group[key]
            if not isinstance(dataset, h5py.Dataset):
                continue
            # Preload the dataset into memory if it fits within the max size limit
            if (
                self.max_in_memory_size_bytes > 0
                and dataset.nbytes <= self.max_in_memory_size_bytes
            ):
                print(f"Preloading dataset {key} of size {dataset.nbytes} bytes into memory.")
                self.cache[key] = torch.from_numpy(dataset[:])
        self._ds_file = None
    def __len__(self):
        return self.length

    @property
    def ds_file(self):
        if self._ds_file is None:
            self._ds_file = h5py.File(self.h5_file_path, "r")
        return self._ds_file
    @property
    def group(self):
        if self.group_name:
            group = self.ds_file[self.group_name]
            if not isinstance(group, h5py.Group):
                raise ValueError(f"Group {self.group_name} is not a valid HDF5 group")
            return group
        else:
            return self.ds_file

    def __getitem__(self, idx: int):
        # nvtx.push_range(f"SparseTensorH5Dataset.__getitem__({idx})")
        # Handle root level access when group_name is empty
        # if self.group_name:
        #     group = self.ds_file[self.group_name]
        #     if not isinstance(group, h5py.Group):
        #         raise ValueError(f"Group {self.group_name} is not a valid HDF5 group")
        # else:
        #     group = self.ds_file  # Use root level

        sample = {}
        for key in self.group.keys():
            if key in self.cache:
                # If the dataset is already cached, use it
                tensor_data = self.cache[key][idx]
            else:
                # Load the dataset from HDF5 file
                dataset = self.group[key]
                if not isinstance(dataset, h5py.Dataset):
                    continue
                #with nvtx.annotate(f"Loading {key} from HDF5"):
                tensor_data = torch.from_numpy(dataset[idx])

            # One-hot encode if key is in num_classes
            if key in self.num_classes:
                # with nvtx.annotate(f"One-hot encoding {key}"):
                sample[key] = torch.nn.functional.one_hot(
                    tensor_data.long(), num_classes=self.num_classes[key]
                ).float()
            else:
                sample[key] = tensor_data
        # nvtx.pop_range()
        return sample
    def get_key(self, idx: int, key: str) -> torch.Tensor:
        if key in self.cache:
            # If the dataset is already cached, use it
            tensor_data = self.cache[key][idx]
        else:
            # Load the dataset from HDF5 file
            dataset = self.group[key]
            if not isinstance(dataset, h5py.Dataset):
                raise ValueError(f"Key {key} is not a valid HDF5 dataset")
            #with nvtx.annotate(f"Loading {key} from HDF5"):
            tensor_data = torch.from_numpy(dataset[idx])

        # One-hot encode if key is in num_classes
        if key in self.num_classes:
            # with nvtx.annotate(f"One-hot encoding {key}"):
            return torch.nn.functional.one_hot(
                tensor_data.long(), num_classes=self.num_classes[key]
            ).float()
        else:
            return tensor_data
        # nvtx.pop_range()