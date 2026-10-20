import torch
from tensordict import (
    TensorDict,  # type: ignore
    assert_allclose_td,  # type: ignore
)

from project.utils.tensor_utils import flatten_tensordict, unflatten_tensor

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


def test_tensordict_flatten():
    example_td = TensorDict(
        {
            "a": torch.tensor([1, 2, 3, 4]),
            "b": torch.tensor([[5, 6], [7, 8], [9, 10], [11, 12]]),
        },
        batch_size=[4],
    )
    example_td = example_td.to(device)
    flattened = flatten_tensordict(example_td)
    assert flattened.shape == (4, 3), "Flattened shape is incorrect"
    unflattened = unflatten_tensor(flattened, {"a": (), "b": (2,)}, num_batchdims=1)
    assert isinstance(unflattened, TensorDict), "Unflattened result is not a TensorDict"
    assert unflattened.batch_size == example_td.batch_size, (
        "Unflattened batch size is incorrect"
    )
    assert assert_allclose_td(unflattened, example_td), (
        "Unflattened TensorDict does not match original"
    )
