from typing import List

import torch
from tensordict import TensorDict  # type: ignore

from project.belief.belief import Belief


class ComposedBelief(Belief):
    """
    A belief that is composed of multiple other beliefs.
    This allows combining different belief types (e.g., Normal and Categorical)
    into a single belief system.
    """

    @classmethod
    def from_config(
        cls,
        beliefs: List[dict],
        device: str | torch.device = "cpu",
    ):
        """
        Create a ComposedBelief from a list of belief configurations.

        Args:
            beliefs: List of belief configurations
            device: The device to use for tensor operations
        """
        belief_objects = []
        for belief_config in beliefs:
            belief_class = belief_config.pop("class")
            belief_object = belief_class.from_config(**belief_config)
            belief_objects.append(belief_object)

        return cls(beliefs=belief_objects, device=device)

    def __init__(
        self,
        beliefs: List[Belief],
        device: str | torch.device = "cpu",
    ):
        """
        Initialize a composed belief from multiple belief objects.

        Args:
            beliefs: List of belief objects to compose
            device: The device to use for tensor operations
        """
        # Combine belief shapes and sample shapes from all component beliefs
        belief_shape = {}
        sample_shape = {}
        for belief in beliefs:
            belief_shape.update(belief.belief_shape)
            sample_shape.update(belief.sample_shape)

        super().__init__(
            belief_shape=belief_shape,
            sample_shape=sample_shape,
        )

        self.beliefs = beliefs
        self.device = torch.device(device)

    def sample_initial_belief(self, batch_size):
        """
        Sample initial beliefs from all component beliefs and combine them.
        """
        if isinstance(batch_size, int):
            batch_size = (batch_size,)

        # Initialize an empty TensorDict with the correct batch size
        combined_belief = TensorDict({}, batch_size=batch_size, device=self.device)

        # Sample from each component belief and add to the combined belief
        for belief in self.beliefs:
            initial_belief = belief.sample_initial_belief(batch_size)
            combined_belief.update(initial_belief)

        return combined_belief

    def sample_from_belief(self, belief):
        """
        Sample from the belief by sampling from each component belief.
        """
        # Initialize empty TensorDict with the same batch size
        samples = TensorDict({}, batch_size=belief.batch_size)

        # Sample from each component belief
        for belief_component in self.beliefs:
            sample = belief_component.sample_from_belief(belief)
            # Add this sample to our result
            samples.update(sample)
            del sample

        return samples

    def sample_belief_update_inplace(self, belief, ground_truth, precision):
        """
        Update the belief inplace based on new observations.
        Delegates to each component belief for the update.
        """
        for belief_component in self.beliefs:
            belief_component.sample_belief_update_inplace(belief, ground_truth, precision)

    def dKL_dt(self, belief, ground_truth, reconstruction, t):
        kl_sum = 0
        for belief_component in self.beliefs:
            kl = belief_component.dKL_dt(belief, ground_truth, reconstruction, t)
            kl_sum += kl
            del kl
        return kl_sum

    def get_precision(self, t):
        """
        Get the precision at time t by collecting precisions from all component beliefs.
        """
        # Initialize empty TensorDict with the same batch size
        precision = TensorDict({}, batch_size=t.shape)

        # Collect precision from each component belief
        for belief_component in self.beliefs:
            component_precision = belief_component.get_precision(t)
            precision.update(component_precision)
            del component_precision

        return precision

    def log_prob(self, belief, sample):
        return sum(
            belief_component.log_prob(belief, sample) for belief_component in self.beliefs
        )

    def to(self, device):
        for belief_component in self.beliefs:
            belief_component.to(device)
        self.device = device

    def get_sde_drift(self, belief, ground_truth):
        drifts = TensorDict({}, batch_size=belief.batch_size, device=self.device)
        for belief_component in self.beliefs:
            drift = belief_component.get_sde_drift(belief, ground_truth)
            drifts.update(drift)
            del drift
        return drifts

    def get_sde_diffusion(self, belief):
        diffusions = TensorDict({}, batch_size=belief.batch_size, device=self.device)
        for belief_component in self.beliefs:
            diffusion = belief_component.get_sde_diffusion(belief)
            diffusions.update(diffusion)
            del diffusion
        return diffusions

    def get_score(self, belief, ground_truth):
        scores = TensorDict({}, batch_size=belief.batch_size, device=self.device)
        for belief_component in self.beliefs:
            score = belief_component.get_score(belief, ground_truth)
            scores.update(score)
            del score
        return scores

    def get_prior_conditioned_score(self, belief, ground_truth, prior):
        scores = TensorDict({}, batch_size=belief.batch_size, device=self.device)
        for belief_component in self.beliefs:
            score = belief_component.get_prior_conditioned_score(
                belief, ground_truth, prior
            )
            scores.update(score)
            del score
        return scores

    def get_flow(self, belief: TensorDict, ground_truth: TensorDict) -> TensorDict:
        flows = TensorDict({}, batch_size=belief.batch_size, device=self.device)
        for belief_component in self.beliefs:
            flow = belief_component.get_flow(belief, ground_truth)
            flows.update(flow)
            del flow
        return flows

    def get_dimentionality(self) -> int:
        return sum(belief.get_dimentionality() for belief in self.beliefs)

    def expand_sample(self, sample: TensorDict) -> TensorDict:
        for belief_component in self.beliefs:
            sample = belief_component.expand_sample(sample)
        return sample

    def discretize_belief_update_inplace(
        self,
        belief: TensorDict,
        ground_truth: TensorDict,
        prior: TensorDict,
        dt: torch.Tensor,
        noise_scale: float | torch.Tensor,
        condition_on_prior: bool,
        method: str = "euler-maruyama",
    ) -> None:
        for belief_component in self.beliefs:
            belief_component.discretize_belief_update_inplace(
                belief, ground_truth, prior, dt, noise_scale, condition_on_prior, method
            )
