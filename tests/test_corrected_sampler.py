import torch
from tensordict import TensorDict  # type: ignore

from project.belief.categorical_belief import CategoricalBelief
from project.prior.softmax_normal_prior import SoftmaxNormalPrior
from project.samplers.step_methods.corrected_variable_noise_step import (
    CorrectedVariableNoiseStep,
    to_simplex,
)
from project.samplers.step_methods.variable_noise_step import VariableNoiseStep
from project.samplers.step_sampler import StepSampler
from project.scheduler.ole_scheduler import OLEScheduler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

KEY = "categorical"
SHAPE = (3,)
NUM_CLASSES = 4


class _Reconstructor(torch.nn.Module):
    """A reconstruction that depends on the belief, so the history matters."""

    def forward(self, belief_params: TensorDict) -> TensorDict:
        logits = 2.0 * belief_params[KEY] + belief_params["t"].view(-1, 1, 1)
        return TensorDict(
            {KEY: torch.softmax(logits, dim=-1)},
            batch_size=belief_params.batch_size,
            device=belief_params.device,
        )


def _belief() -> CategoricalBelief:
    return CategoricalBelief(
        shape=SHAPE,
        num_classes=NUM_CLASSES,
        categorical_prior=SoftmaxNormalPrior(
            shape=SHAPE, num_classes=NUM_CLASSES, device=device, initial_belief_std=1.0
        ),
        scheduler=OLEScheduler(initial_precision=1.0, final_precision=10.0),
        key=KEY,
        device=device,
    )


def _sampler(step_method) -> StepSampler:
    return StepSampler(
        name="test",
        step_method=step_method,
        belief=_belief(),
        reconstructor=_Reconstructor().to(device),
        device=device,
        max_function_evals=8,
    )


def _sample(step_method, num_batches: int = 1, batch_size: int = 4) -> list[torch.Tensor]:
    sampler = _sampler(step_method)
    out = []
    for i in range(num_batches):
        torch.manual_seed(i)
        sample, _ = sampler.sample_batch(batch_size)
        out.append(sample[KEY].clone())
    return out


def test_corrections_off_reproduce_the_released_step():
    """The corrected step method must be a no-op until a weight is set."""
    released = _sample(VariableNoiseStep(noise_scale=2.0, condition_on_prior=False))
    corrected = _sample(
        CorrectedVariableNoiseStep(noise_scale=2.0, condition_on_prior=False)
    )
    assert torch.equal(released[0], corrected[0])


def test_corrections_change_the_samples():
    """...and must do something once one is set, or the test above is vacuous."""
    released = _sample(VariableNoiseStep(noise_scale=2.0, condition_on_prior=False))
    for kwargs in ({"extrapolation": 1.0}, {"guidance": 0.5}):
        corrected = _sample(
            CorrectedVariableNoiseStep(
                noise_scale=2.0, condition_on_prior=False, **kwargs
            )
        )
        assert not torch.allclose(released[0], corrected[0]), kwargs


def test_guidance_formula():
    step = CorrectedVariableNoiseStep(
        noise_scale=2.0, condition_on_prior=False, guidance=0.5
    )
    belief_params = _belief().sample_initial_belief(5)
    belief_params["t"] = torch.zeros(5, device=device)
    reconstruction = _Reconstructor().to(device)(belief_params)
    before = reconstruction[KEY].clone()

    step.correct_reconstruction(
        reconstruction, belief_params, [KEY], torch.tensor(0.1, device=device)
    )

    expected = to_simplex(before + 0.5 * (before - torch.softmax(belief_params[KEY], -1)))
    assert torch.allclose(reconstruction[KEY], expected)


def test_extrapolation_through_the_previous_reconstruction():
    step = CorrectedVariableNoiseStep(
        noise_scale=2.0, condition_on_prior=False, extrapolation=2.0, extrapolation_clip=False
    )
    belief_params = _belief().sample_initial_belief(5)
    first = torch.rand(5, *SHAPE, NUM_CLASSES, device=device)
    second = torch.rand(5, *SHAPE, NUM_CLASSES, device=device)
    dt_1, dt_2 = 0.2, 0.1

    held = TensorDict({KEY: first.clone()}, batch_size=(5,), device=device)
    step.correct_reconstruction(
        held, belief_params, [KEY], torch.tensor(dt_1, device=device)
    )
    # The first step has no history, so it passes the reconstruction through.
    assert torch.equal(held[KEY], first)

    held = TensorDict({KEY: second.clone()}, batch_size=(5,), device=device)
    step.correct_reconstruction(
        held, belief_params, [KEY], torch.tensor(dt_2, device=device)
    )
    expected = second + 2.0 * (dt_2 / dt_1) * (second - first)
    assert torch.allclose(held[KEY], expected)


def test_extrapolation_clip_projects_onto_the_simplex():
    """Clipping cannot zero out a whole row.

    The reconstructions the step method is given are probability vectors, so the
    extrapolation `(1 + w) x_n - w x_{n-1}` sums to one before clipping and at
    least one of its entries is therefore positive, however large `w` is. That
    is what keeps `to_simplex` from dividing by zero.
    """
    step = CorrectedVariableNoiseStep(
        noise_scale=2.0, condition_on_prior=False, extrapolation=5.0, extrapolation_clip=True
    )
    belief_params = _belief().sample_initial_belief(5)
    for _ in range(2):
        probabilities = torch.softmax(
            torch.randn(5, *SHAPE, NUM_CLASSES, device=device), dim=-1
        )
        held = TensorDict({KEY: probabilities}, batch_size=(5,), device=device)
        step.correct_reconstruction(
            held, belief_params, [KEY], torch.tensor(0.1, device=device)
        )
    assert torch.all(held[KEY] >= 0)
    assert torch.allclose(held[KEY].sum(-1), torch.ones_like(held[KEY].sum(-1)))


def test_history_does_not_leak_between_batches():
    """Two batches drawn with the same seed must come out identical."""
    step = CorrectedVariableNoiseStep(
        noise_scale=2.0, condition_on_prior=False, extrapolation=1.0, guidance=0.3
    )
    first, second = _sample(step, num_batches=2)
    torch.manual_seed(1)
    assert step._previous is not None  # the history was built during the batch
    assert torch.equal(first, _sample(step)[0])
    assert not torch.equal(first, second)  # different seeds, different samples


def test_corrections_need_the_smooth_reconstruction_mode():
    for kwargs in ({"extrapolation": 1.0}, {"guidance": 0.5}):
        try:
            CorrectedVariableNoiseStep(
                noise_scale=2.0,
                condition_on_prior=False,
                reconstruction_mode="mode",
                **kwargs,
            )
        except ValueError:
            continue
        raise AssertionError(f"{kwargs} was accepted with reconstruction_mode='mode'")
