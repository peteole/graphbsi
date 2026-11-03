"""Corrections for the posterior-mean bias of the reconstructor.

The reconstructor predicts a posterior mean: an average over every graph that is
consistent with the current noisy belief. That average lies between real graphs,
so repeatedly steering the belief at it biases samples towards generic,
mode-averaged structures. The two corrections below push the target further
along the direction the reconstructor already proposes. Both reuse quantities
that are already computed, so neither costs a function evaluation:

- ``guidance``: feed the belief update ``x + w * (x - softmax(belief))``, which
  over-relaxes the update direction.
- ``extrapolation``: feed it a linear extrapolation through the previous
  reconstruction, ``x_n + w * dt * (x_n - x_{n-1}) / dt_prev``.

Nothing here touches the released sampler: this is a separate step method,
selected by its own config, and :class:`VariableNoiseStep` is used unchanged
when both weights are zero.
"""

from typing import Literal, Optional

import torch
from tensordict import TensorDict  # type: ignore

from project.belief.belief import Belief
from project.reconstructor.reconstructor import SampleReconstructor
from project.samplers.step_methods.variable_noise_step import (
    VariableNoiseStep,
    get_categorical_belief_keys,
)


def to_simplex(probabilities: torch.Tensor) -> torch.Tensor:
    """Project onto the probability simplex by clipping and renormalizing."""
    probabilities = probabilities.clamp_min(0)
    return probabilities / probabilities.sum(dim=-1, keepdim=True).clamp_min(1e-12)


class CorrectedVariableNoiseStep(VariableNoiseStep):
    def __init__(
        self,
        noise_scale: float,
        condition_on_prior: bool,
        method: Literal["euler-maruyama", "ornstein-uhlenbeck"] = "euler-maruyama",
        bayesian_till: float = 0.0,
        reconstruction_mode: Literal["smooth", "mode", "sample"] = "smooth",
        extrapolation: float = 0.0,
        extrapolation_clip: bool = True,
        guidance: float = 0.0,
    ):
        """
        Args:
            extrapolation: weight of the multistep correction, which replaces the
                reconstruction by a linear extrapolation through the previous
                one, ``x_n + w * dt * (x_n - x_{n-1}) / dt_prev``. A weight of
                0.5 is the second-order (Adams-Bashforth) scheme; larger values
                overshoot deliberately and work better in practice.
            extrapolation_clip: project the extrapolated reconstruction back onto
                the simplex. Leaving it off is better at few function
                evaluations.
            guidance: weight of the belief guidance correction,
                ``x + w * (x - softmax(belief))``. Helps where the time
                discretization is coarse, hurts where it is fine.

        The remaining arguments are those of :class:`VariableNoiseStep` and keep
        their meaning. With ``extrapolation=0`` and ``guidance=0`` this step is
        the released one.
        """
        super().__init__(
            noise_scale=noise_scale,
            condition_on_prior=condition_on_prior,
            method=method,
            bayesian_till=bayesian_till,
            reconstruction_mode=reconstruction_mode,
        )
        if (extrapolation != 0.0 or guidance != 0.0) and reconstruction_mode != "smooth":
            raise ValueError(
                "The reconstruction corrections operate on probabilities and "
                f"need reconstruction_mode='smooth', got {reconstruction_mode!r}"
            )
        self.extrapolation = extrapolation
        self.extrapolation_clip = extrapolation_clip
        self.guidance = guidance
        self.reset()

    @property
    def corrects(self) -> bool:
        return self.extrapolation != 0.0 or self.guidance != 0.0

    def reset(self) -> None:
        """Forget the previous reconstruction."""
        self._previous: Optional[dict[str, torch.Tensor]] = None
        self._previous_dt: Optional[float] = None

    def step(
        self,
        belief: Belief,
        belief_params: TensorDict,
        reconstructor: SampleReconstructor,
        dt: torch.Tensor,
        prior_params: TensorDict,
    ) -> None:
        if not self.corrects:
            return super().step(belief, belief_params, reconstructor, dt, prior_params)
        # The multistep term remembers the previous reconstruction, which must
        # not leak across batches. Every batch starts its time grid at t = 0.
        if bool(torch.all(belief_params["t"] == 0.0)):
            self.reset()
        corrected = _CorrectingReconstructor(
            reconstructor, self, get_categorical_belief_keys(belief), dt
        )
        super().step(belief, belief_params, corrected, dt, prior_params)

    def correct_reconstruction(
        self,
        reconstruction: TensorDict,
        belief_params: TensorDict,
        keys: list[str],
        dt: torch.Tensor,
    ) -> None:
        """Apply belief guidance and multistep extrapolation in place."""
        if self.guidance != 0.0:
            for key in keys:
                belief_estimate = torch.softmax(belief_params[key], dim=-1)
                reconstruction[key] = to_simplex(
                    reconstruction[key]
                    + self.guidance * (reconstruction[key] - belief_estimate)
                )
        if self.extrapolation == 0.0:
            return
        current = {key: reconstruction[key].clone() for key in keys}
        usable = (
            self._previous is not None
            and self._previous_dt is not None
            and all(
                self._previous[key].shape == reconstruction[key].shape for key in keys
            )
        )
        if usable:
            assert self._previous is not None and self._previous_dt is not None
            scale = self.extrapolation * float(dt) / self._previous_dt
            for key in keys:
                extrapolated = reconstruction[key] + scale * (
                    reconstruction[key] - self._previous[key]
                )
                if self.extrapolation_clip:
                    extrapolated = to_simplex(extrapolated)
                reconstruction[key] = extrapolated
        self._previous = current
        self._previous_dt = float(dt)


class _CorrectingReconstructor:
    """Corrects whatever the wrapped reconstructor returns.

    Wrapping the reconstructor rather than copying the body of
    :meth:`VariableNoiseStep.step` keeps the correction out of the released step
    method while still applying it in the right place: after the reconstruction,
    before the belief update.
    """

    def __init__(
        self,
        reconstructor: SampleReconstructor,
        step_method: CorrectedVariableNoiseStep,
        keys: list[str],
        dt: torch.Tensor,
    ):
        self._reconstructor = reconstructor
        self._step_method = step_method
        self._keys = keys
        self._dt = dt

    def __call__(self, belief_params: TensorDict) -> TensorDict:
        reconstruction = self._reconstructor(belief_params)
        self._step_method.correct_reconstruction(
            reconstruction, belief_params, self._keys, self._dt
        )
        return reconstruction

    def __getattr__(self, name: str):
        return getattr(self._reconstructor, name)
