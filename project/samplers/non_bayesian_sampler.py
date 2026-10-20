from itertools import pairwise
from typing import Literal, Optional

import torch
from tensordict import TensorDict  # type: ignore

from project.belief.belief import Belief
from project.belief.categorical_belief import CategoricalBelief
from project.belief.composed_belief import ComposedBelief
from project.class_conditioner.class_conditioner import ClassConditioner
from project.reconstructor.reconstructor import SampleReconstructor
from project.samplers.sampler import Sampler
from torch.distributions import Categorical

def get_categorical_belief_keys(belief: Belief)-> list[str]:
    if isinstance(belief, CategoricalBelief):
        return [belief.key]
    if isinstance(belief, ComposedBelief):
        keys = []
        for b in belief.beliefs:
            keys.extend(get_categorical_belief_keys(b))
        return keys
    return []

class NonBayesianSampler(Sampler):
    def __init__(
        self,
        name: str,
        belief: Belief,
        reconstructor: SampleReconstructor,
        class_conditioner: Optional[ClassConditioner] = None,
        device: torch.device = torch.device("cpu"),
        rho: float = 1.0,
        t_max: float = 1.0,
        time_discretization_steps: int = 100,
        resample_initial_belief: bool = True,
        reconstruction_mode: Literal["smooth", "mode", "sample"] = "smooth"
    ):
        super().__init__(name, belief, reconstructor, class_conditioner, device)
        self.timesteps = self.get_timesteps(
            t_min=0,
            t_max=t_max,
            rho=rho,
            num_discretization_steps=time_discretization_steps,
        )
        self.resample_initial_belief = resample_initial_belief
        self.reconstruction_mode = reconstruction_mode
        self.categorical_belief_keys = get_categorical_belief_keys(belief)

    def sample_batch(
        self,
        batch_size,
        belief_timesteps_to_save=[],
        conditioner=None,
    ):
        with torch.no_grad():
            belief_series: list[TensorDict] = []
            belief = self.belief.sample_initial_belief(batch_size)
            initial_belief = belief.clone() if not self.resample_initial_belief else None
            belief["t"] = torch.zeros(batch_size, device=self.device)
            classifications = None
            if self.class_conditioner is not None:
                classifications = self.class_conditioner.sample_classifications(
                    batch_size, device=self.device
                )
                self.class_conditioner.write_sample_classifications_to_belief(
                    classifications, belief
                )
            belief_timesteps_to_save = sorted(belief_timesteps_to_save)
            belief_timesteps_save_idx = 0
            self.reconstructor.to(self.device)
            if 0 in belief_timesteps_to_save:
                # If the first timestep to save is 0, we save the initial belief
                belief_series.append(belief.clone())
                belief_timesteps_save_idx += 1

            final_sample = None
            for t1, t2 in pairwise(self.timesteps):
                t = torch.full((batch_size,), t1, device=self.device)
                reconstruction = self.reconstructor(belief)
                if self.reconstruction_mode == "smooth":
                    pass
                elif self.reconstruction_mode == "mode":
                    for key in self.categorical_belief_keys:
                        reconstruction[key] = reconstruction[key].argmax(dim=-1)
                elif self.reconstruction_mode == "sample":
                    for key in self.categorical_belief_keys:
                        probs = reconstruction[key]
                        dist = Categorical(probs)
                        reconstruction[key] = dist.sample()
                else:
                    raise ValueError(f"Invalid reconstruction mode: {self.reconstruction_mode}")
                precision = self.belief.get_precision(t)
                del belief
                belief = self.belief.sample_initial_belief(batch_size) if initial_belief is None else initial_belief.clone()
                self.belief.sample_belief_update_inplace(
                    belief, reconstruction, precision
                )
                belief["t"] = t
                if conditioner is not None:
                    conditioner(reconstruction)
                if self.class_conditioner is not None and classifications is not None:
                    self.class_conditioner.write_sample_classifications_to_belief(
                        classifications, belief
                    )
                # Clear intermediate tensors
                del reconstruction, t, precision
                while (
                    belief_timesteps_save_idx < len(belief_timesteps_to_save)
                    and t1 <= belief_timesteps_to_save[belief_timesteps_save_idx] <= t2
                ):
                    # Add beliefs to the series for all timesteps to save in [t, t + dt]. Normally this would only include one timestep but the while loop ensures that the number of saved beliefs is equal to the number of timesteps to save.
                    belief_series.append(belief.clone())
                    belief_timesteps_save_idx += 1
            while belief_timesteps_save_idx < len(belief_timesteps_to_save):
                # If there are still timesteps to save, we save the final belief
                belief_series.append(belief.clone())
                belief_timesteps_save_idx += 1
            final_sample = self.reconstructor(belief)
            if self.class_conditioner is not None and classifications is not None:
                self.class_conditioner.write_sample_classifications_to_belief(
                    classifications, final_sample
                )
            del belief
            return final_sample, belief_series
