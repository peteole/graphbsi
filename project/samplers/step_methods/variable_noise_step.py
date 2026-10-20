from typing import Literal
import torch
from project.belief.belief import Belief
from project.belief.categorical_belief import CategoricalBelief
from project.belief.composed_belief import ComposedBelief
from project.reconstructor.reconstructor import SampleReconstructor

from project.samplers.step_methods.step_method import StepMethod
from tensordict import TensorDict
from torch.distributions import Categorical


def get_categorical_belief_keys(belief: Belief) -> list[str]:
    if isinstance(belief, CategoricalBelief):
        return [belief.key]
    if isinstance(belief, ComposedBelief):
        keys = []
        for b in belief.beliefs:
            keys.extend(get_categorical_belief_keys(b))
        return keys
    return []

class VariableNoiseStep(StepMethod):
    def __init__(
        self,
        noise_scale: float,
        condition_on_prior: bool,
        method: Literal["euler-maruyama", "ornstein-uhlenbeck"] = "euler-maruyama",
        bayesian_till: float = 0.0,
        reconstruction_mode: Literal["smooth", "mode", "sample"] = "smooth",
    ):
        """
        Initialize the VariableNoiseStep with a noise scale. noise_scale=0 is an ODE, noise_scale=1 is the SDE, and noise_scale>1 is even more noisy.
        """
        self.noise_scale = noise_scale
        self.condition_on_prior = condition_on_prior
        self.method = method
        self.bayesian_till = bayesian_till
        self.reconstruction_mode = reconstruction_mode

    def step(
        self,
        belief: Belief,
        belief_params: TensorDict,
        reconstructor: SampleReconstructor,
        dt: torch.Tensor,
        prior_params: TensorDict,
    ) -> None:
        reconstruction = reconstructor(belief_params)
        categorical_belief_keys = get_categorical_belief_keys(belief)
        if self.reconstruction_mode == "smooth":
            pass
        elif self.reconstruction_mode == "mode":
            for key in categorical_belief_keys:
                reconstruction[key] = reconstruction[key].argmax(dim=-1)
        elif self.reconstruction_mode == "sample":
            for key in categorical_belief_keys:
                probs = reconstruction[key]
                dist = Categorical(probs)
                reconstruction[key] = dist.sample()
        else:
            raise ValueError(f"Invalid reconstruction mode: {self.reconstruction_mode}")
        t = belief_params["t"]
        if torch.any(t < self.bayesian_till):
            precision_increment = belief.get_precision(t + dt) - belief.get_precision(t)
            belief.sample_belief_update_inplace(
                belief_params, reconstruction, precision_increment
            )
            return
        belief.discretize_belief_update_inplace(
            belief_params,
            reconstruction,
            prior_params,
            dt,
            self.noise_scale,
            self.condition_on_prior,
            method=self.method,
        )
