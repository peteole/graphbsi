from itertools import pairwise
from typing import Optional

import torch
from tensordict import TensorDict  # type: ignore

from project.belief.belief import Belief
from project.class_conditioner.class_conditioner import ClassConditioner
from project.reconstructor.reconstructor import SampleReconstructor
from project.samplers.sampler import Sampler


class BayesianSampler(Sampler):
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
    ):
        super().__init__(name, belief, reconstructor, class_conditioner, device)
        self.timesteps = self.get_timesteps(
            t_min=0,
            t_max=t_max,
            rho=rho,
            num_discretization_steps=time_discretization_steps,
        )

    def sample_batch(
        self,
        batch_size,
        belief_timesteps_to_save=[],
        conditioner=None,
    ):
        with torch.no_grad():
            belief_series: list[TensorDict] = []
            belief = self.belief.sample_initial_belief(batch_size)
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
            if 0 in belief_timesteps_to_save:
                # If the first timestep to save is 0, we save the initial belief
                belief_series.append(belief.clone())
                belief_timesteps_save_idx += 1

            self.reconstructor.to(self.device)
            final_sample = None
            for (t1, t2) in pairwise(self.timesteps):
                t = torch.full((batch_size,), t1, device=self.device)
                belief["t"] = t
                reconstruction = self.reconstructor(belief)
                if conditioner is not None:
                    conditioner(reconstruction)
                precision_increment = self.belief.get_precision(
                    t2.to(self.device)
                ) - self.belief.get_precision(t1.to(self.device))
                self.belief.sample_belief_update_inplace(
                    belief, reconstruction, precision_increment
                )
                # Clear intermediate tensors
                del reconstruction, t, precision_increment
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
