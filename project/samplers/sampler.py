from abc import abstractmethod
from typing import Optional

import torch
from tensordict import TensorDict
from tqdm import tqdm

from project.belief.belief import Belief
from project.class_conditioner.class_conditioner import ClassConditioner
from project.conditioner.conditioner import Conditioner
from project.reconstructor.reconstructor import SampleReconstructor


class Sampler:
    def __init__(
        self,
        name: str,
        belief: Belief,
        reconstructor: SampleReconstructor,
        class_conditioner: Optional[ClassConditioner] = None,
        device: torch.device = torch.device("cpu"),
    ):
        self.name = name
        self.belief = belief
        self.reconstructor = reconstructor
        self.class_conditioner = class_conditioner
        self.device = device

    @abstractmethod
    def sample_batch(
        self,
        batch_size: int,
        belief_timesteps_to_save: list[float] = [],
        conditioner: Optional[Conditioner] = None,
    ) -> tuple[TensorDict, list[TensorDict]]:
        """
        Sample a batch from the model.
        Args:
            batch_size (int): The number of samples to generate.
            belief_timesteps_to_save (list[float]): A list of time steps at which to save the belief state.
            conditioner (Callable[[TensorDict],None]): A function that modifies the belief reconstruction in place for conditioning. May be used to fix some features of the reconstruction. If None, no conditioning is applied.
        Returns:
            TensorDict or tuple[list[TensorDict], Tensordict]: A batch of samples from the model. If `return_belief_series` is True, returns a list of beliefs at each time step and the final sample.
        """
        pass

    def sample(
        self,
        num_samples: int,
        batch_size: int,
        belief_timesteps_to_save: list[float] = [],
        conditioner: Optional[Conditioner] = None,
    ) -> tuple[TensorDict, list[TensorDict]]:
        """
        Sample from the model by repeatedly calling `sample_batch` until the desired number of samples is reached.
        Args:
            num_samples (int): The number of samples to generate.
            batch_size (int): The number of samples to generate in each call to `sample_batch`.
            belief_timesteps_to_save (list[float]): A list of time steps at which to save the belief state.
            conditioner (Callable[[TensorDict],None]): A function that modifies the belief reconstruction in place for conditioning. May be used to fix some features of the reconstruction. If None, no conditioning is applied.
        Returns:
            TensorDict or tuple[list[TensorDict], Tensordict]: A batch of samples from the model. If `return_belief_series` is True, returns a list of beliefs at each time step and the final sample.
        """
        sample_batches = []
        belief_batches = []
        num_samples_generated = 0
        with tqdm(total=num_samples, desc=f"Sampling with {self.name}") as pbar:
            while num_samples_generated < num_samples:
                batch_size_to_generate = min(
                    batch_size, num_samples - num_samples_generated
                )
                belief_batch, belief_series = self.sample_batch(
                    batch_size=batch_size_to_generate,
                    belief_timesteps_to_save=belief_timesteps_to_save,
                    conditioner=conditioner,
                )
                sample_batches.append(belief_batch)
                belief_batches.append(belief_series)
                num_samples_generated += batch_size_to_generate
                pbar.update(batch_size_to_generate)

        samples: TensorDict = TensorDict.cat(sample_batches, dim=0)  # type: ignore
        beliefs_series: list[TensorDict] = [
            TensorDict.cat([bb[i] for bb in belief_batches], dim=0)
            for i in range(len(belief_timesteps_to_save))
        ]  # type: ignore
        return samples, beliefs_series

    def get_timesteps(
        self,
        t_min: float = 0.0,
        t_max: float = 1.0,
        rho: float = 1.0,
        num_discretization_steps: int = 500,
    ) -> torch.Tensor:
        t = torch.linspace(0, 1, num_discretization_steps)
        t_s = (t_min ** (1 / rho) + t * (t_max ** (1 / rho) - t_min ** (1 / rho))) ** rho
        return t_s

    def to(self, device: torch.device) -> None:
        """
        Move the sampler to a device.
        Args:
            device (torch.device): The device to move the sampler to.
        """
        self.device = device
        self.belief.to(device)
        self.reconstructor = self.reconstructor.to(device)
