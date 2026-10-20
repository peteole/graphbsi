from typing import Literal, Protocol
import torch
from project.belief.belief import Belief
from project.reconstructor.reconstructor import SampleReconstructor
from project.samplers.step_methods.step_method import StepMethod
from tensordict import TensorDict


class ODERhs(Protocol):
    def __call__(
        self,
        belief: Belief,
        belief_params: TensorDict,
        reconstruction: TensorDict,
        t: torch.Tensor,
    ) -> TensorDict:
        """Evaluate the right-hand side of the ODE. The stepper will update d belief_params = rhs(belief_params, reconstruction, t) dt"""
        ...


class ScoreMatchingODERhs(ODERhs):
    def __call__(
        self,
        belief: Belief,
        belief_params: TensorDict,
        reconstruction: TensorDict,
        t: torch.Tensor,
    ) -> TensorDict:
        belief_params["t"] = t
        drift = belief.get_sde_drift(belief_params, reconstruction)
        diffusion = belief.get_sde_diffusion(belief_params)
        score = belief.get_score(belief_params, reconstruction)
        result = drift - 0.5 * diffusion**2 * score  # type: ignore
        return result

class FlowMatchingRhs(ODERhs):
    def __call__(self, belief: Belief, belief_params: TensorDict, reconstruction: TensorDict, t: torch.Tensor) -> TensorDict:
        belief_params["t"] = t
        return belief.get_flow(belief_params, ground_truth=reconstruction)


class EulerStep(StepMethod):
    """A simple Euler step method for ODEs."""
    
    @classmethod
    def from_config(cls, rhs: Literal["score_matching", "flow_matching"], **kwargs) -> "EulerStep":
        if rhs == "score_matching":
            rhs_instance = ScoreMatchingODERhs()
        elif rhs == "flow_matching":
            rhs_instance = FlowMatchingRhs()
        else:
            raise ValueError(f"Unknown RHS type: {rhs}")
        return cls(rhs_instance)

    def __init__(self, rhs: ODERhs) -> None:
        super().__init__()
        self.rhs = rhs

    def step(
        self,
        belief: Belief,
        belief_params: TensorDict,
        reconstructor: SampleReconstructor,
        dt: torch.Tensor,
        prior_params: TensorDict,
    ) -> None:
        # Update the belief_params in-place to t = belief["t"] + dt
        reconstruction = reconstructor(belief_params)
        rhs_result = self.rhs(belief, belief_params, reconstruction, belief_params["t"])
        for key in rhs_result.keys():
            belief_params[key] = belief_params[key] + rhs_result[key] * dt


class HeunStep(StepMethod):
    """A Heun step method for ODEs."""

    def __init__(self, rhs: ODERhs) -> None:
        super().__init__()
        self.rhs = rhs

    def step(
        self,
        belief: Belief,
        belief_params: TensorDict,
        reconstructor: SampleReconstructor,
        dt: torch.Tensor,
        prior_params: TensorDict,
    ) -> None:
        # perform 2nd order Runge-Kutta step
        k1 = self.rhs(belief, belief_params, reconstructor(belief_params), belief_params["t"])
        belief_params_k1 = belief_params.clone()
        for key in k1.keys():
            belief_params_k1[key] = belief_params[key] + 0.5 * dt * k1[key]
        k2 = self.rhs(belief, belief_params_k1, reconstructor(belief_params_k1), belief_params["t"] + 0.5 * dt)
        for key in k2.keys():
            belief_params[key] = belief_params[key] + dt * k2[key]
        del belief_params_k1, k1, k2
    @property
    def num_function_evals(self) -> int:
        return 2


