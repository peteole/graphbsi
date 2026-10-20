import torch
from tensordict import TensorDict  # type: ignore

from project.reconstructor.reconstructor import SampleReconstructor


class MockReconstructor(SampleReconstructor):
    def __init__(self, reconstruction: TensorDict):
        super().__init__()
        self.reconstruction = reconstruction

    def forward(self, belief: TensorDict) -> TensorDict:
        num_samples = belief.batch_size[0]
        repeated_reconstruction: TensorDict = torch.cat(
            [
                self.reconstruction
                for _ in range(num_samples // self.reconstruction.batch_size[0] + 1)
            ]
        )  # type: ignore
        return repeated_reconstruction[:num_samples]
