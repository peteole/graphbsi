from itertools import pairwise
from typing import Literal, Optional

import torch
from tensordict import TensorDict  # type: ignore

from project.belief.belief import Belief
from project.class_conditioner.class_conditioner import ClassConditioner
from project.reconstructor.reconstructor import SampleReconstructor
from project.samplers.sampler import Sampler
from project.samplers.step_methods.bayesian import BayesianStep
from project.samplers.step_methods.ode_step import EulerStep, FlowMatchingRhs, HeunStep, ScoreMatchingODERhs
from project.samplers.step_methods.step_method import StepMethod


class StepSampler(Sampler):
    @classmethod
    def from_config(
        cls,
        sampler_type: Literal["bayesian", "flow_matching_euler", "flow_matching_heun","score_matching_euler","score_matching_heun"],
        name: str,
        belief: Belief,
        reconstructor: SampleReconstructor,
        class_conditioner: Optional[ClassConditioner] = None,
        device: torch.device = torch.device("cpu"),
        rho: float = 1.0,
        t_max: float = 1.0,
        max_function_evals: int = 100,
    ):
        if sampler_type == "bayesian":
            step_method = BayesianStep()
        elif sampler_type == "flow_matching_euler":
            step_method = EulerStep(FlowMatchingRhs())
        elif sampler_type == "flow_matching_heun":
            step_method = HeunStep(FlowMatchingRhs())
        elif sampler_type == "score_matching_euler":
            step_method = EulerStep(ScoreMatchingODERhs())
        elif sampler_type == "score_matching_heun":
            step_method = HeunStep(ScoreMatchingODERhs())
        else:
            raise ValueError(f"Unknown sampler type: {sampler_type}")

        return cls(
            name=name,
            step_method=step_method,
            belief=belief,
            reconstructor=reconstructor,
            class_conditioner=class_conditioner,
            device=device,
            rho=rho,
            t_max=t_max,
            max_function_evals=max_function_evals,
        )

    def __init__(
        self,
        name: str,
        step_method: StepMethod,
        belief: Belief,
        reconstructor: SampleReconstructor,
        class_conditioner: Optional[ClassConditioner] = None,
        device: torch.device = torch.device("cpu"),
        rho: float = 1.0,
        t_max: float = 1.0,
        max_function_evals: int = 100,
    ):
        super().__init__(name, belief, reconstructor, class_conditioner, device)
        num_time_discretization_steps = (
            max_function_evals - 1
        ) // step_method.num_function_evals  # one final evaluation
        self.timesteps = self.get_timesteps(
            t_min=0,
            t_max=t_max,
            rho=rho,
            num_discretization_steps=num_time_discretization_steps,
        )
        self.step_method = step_method

    def sample_batch(
        self,
        batch_size,
        belief_timesteps_to_save=[],
        conditioner=None,
    ):
        assert conditioner is None, "Conditioner is not supported in StepSampler"
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
            prior_params = belief.clone()
            belief_timesteps_to_save = sorted(belief_timesteps_to_save)
            belief_timesteps_save_idx = 0
            if 0 in belief_timesteps_to_save:
                # If the first timestep to save is 0, we save the initial belief
                belief_series.append(belief.clone())
                belief_timesteps_save_idx += 1

            self.reconstructor.to(self.device)
            for t1, t2 in pairwise(self.timesteps):
                t = torch.full((batch_size,), t1.cpu().item(), device=self.device)
                belief["t"] = t
                self.step_method.step(
                    self.belief,
                    belief,
                    self.reconstructor,
                    (t2 - t1).to(self.device),
                    prior_params
                )
                # Clear intermediate tensors
                del t
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
