import torch
from project.belief.belief import Belief
from project.reconstructor.reconstructor import SampleReconstructor
from project.samplers.step_methods.step_method import StepMethod
from tensordict import TensorDict


class BayesianStep(StepMethod):
    def step(
        self,
        belief: Belief,
        belief_params: TensorDict,
        reconstructor: SampleReconstructor,
        dt: torch.Tensor,
    ) -> None:
        # Update the belief_params in-place to t = belief["t"] + dt
        reconstruction = reconstructor(belief_params)
        precision_increment = belief.get_precision(
            belief_params["t"] + dt
        ) - belief.get_precision(belief_params["t"])
        belief_params["t"] = belief_params["t"] + dt
        belief.sample_belief_update_inplace(
            belief_params, reconstruction, precision_increment
        )
        del precision_increment, reconstruction
