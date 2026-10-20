
from abc import abstractmethod
from typing import Protocol

from project.belief.belief import Belief
from tensordict import TensorDict
import torch

from project.reconstructor.reconstructor import SampleReconstructor

class StepMethod(Protocol):
    @abstractmethod
    def step(
        self,
        belief: Belief,
        belief_params: TensorDict,
        reconstructor: SampleReconstructor,
        dt: torch.Tensor,
        prior_params: TensorDict,
    ) -> None:
        """Perform a single step of the sampling process. Updates the belief_params in-place to t = belief_params["t"]+dt"""
        ...
    @property
    def num_function_evals(self) -> int:
        """Return the number of function evaluations performed by this step method."""
        return 1