from itertools import pairwise
from typing import Literal, Optional

import torch
from tensordict import TensorDict  # type: ignore
from torch.distributions import Categorical

from project.belief.belief import Belief
from project.belief.categorical_belief import CategoricalBelief
from project.belief.composed_belief import ComposedBelief
from project.class_conditioner.class_conditioner import ClassConditioner
from project.reconstructor.reconstructor import SampleReconstructor
from project.samplers.sampler import Sampler
from project.utils.tensor_utils import flatten_tensordict


def get_categorical_belief_keys(belief: Belief)-> list[str]:
    if isinstance(belief, CategoricalBelief):
        return [belief.key]
    if isinstance(belief, ComposedBelief):
        keys = []
        for b in belief.beliefs:
            keys.extend(get_categorical_belief_keys(b))
        return keys
    return []

class FlowBackSampler(Sampler):
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
        reconstruction_mode: Literal["smooth", "mode", "sample"] = "smooth",
        flowback_threshold: Optional[float] = 0.5,
        flowback_min_step: int = 2,
        flowback_variance_floor: float = 1e-4,
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
        self.flowback_threshold = flowback_threshold
        self.flowback_min_step = max(flowback_min_step, 0)
        self.flowback_variance_floor = flowback_variance_floor

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
            previous_raw_reconstruction: Optional[TensorDict] = None
            for step_index, (t1, t2) in enumerate(pairwise(self.timesteps)):
                t = torch.full((batch_size,), t1, device=self.device)
                next_t = torch.full((batch_size,), t2, device=self.device)
                raw_reconstruction = self.reconstructor(belief)
                reconstruction = (
                    raw_reconstruction
                    if self.reconstruction_mode == "smooth"
                    else raw_reconstruction.clone()
                )
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
                use_flowback = self._should_apply_flowback(
                    step_index, raw_reconstruction, previous_raw_reconstruction
                )
                if use_flowback:
                    del belief
                    belief = (
                        self.belief.sample_initial_belief(batch_size)
                        if initial_belief is None
                        else initial_belief.clone()
                    )
                    self.belief.sample_belief_update_inplace(
                        belief, reconstruction, precision
                    )
                else:
                    precision_increment = self.belief.get_precision(
                        t2.to(self.device)
                    ) - self.belief.get_precision(t1.to(self.device))
                    self.belief.sample_belief_update_inplace(
                        belief, reconstruction, precision_increment
                    )
                belief["t"] = t
                if conditioner is not None:
                    conditioner(reconstruction)
                if self.class_conditioner is not None and classifications is not None:
                    self.class_conditioner.write_sample_classifications_to_belief(
                        classifications, belief
                    )
                # Clear intermediate tensors
                if self.flowback_threshold is not None:
                    previous_raw_reconstruction = raw_reconstruction.detach().clone()
                del reconstruction, raw_reconstruction, t, next_t, precision
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

    def _should_apply_flowback(
        self,
        step_index: int,
        current_reconstruction: TensorDict,
        previous_reconstruction: Optional[TensorDict],
    ) -> bool:
        if (
            self.flowback_threshold is None
            or not self.categorical_belief_keys
            or previous_reconstruction is None
            or step_index < self.flowback_min_step
        ):
            return False
        current_flat = flatten_tensordict(current_reconstruction, ignore_keys={"t"})
        previous_flat = flatten_tensordict(previous_reconstruction, ignore_keys={"t"})
        if current_flat.shape != previous_flat.shape:
            return False
        per_sample_norm = torch.linalg.vector_norm(
            current_flat - previous_flat, dim=-1
        )
        mean_norm = per_sample_norm.mean()
        return bool(mean_norm < self.flowback_threshold)
