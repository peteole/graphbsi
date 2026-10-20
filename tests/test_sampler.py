from typing import Any

import torch
from tensordict import (
    TensorDict,  # type: ignore
    assert_allclose_td,
)
from torch.nn.functional import one_hot

from project.belief.categorical_belief import CategoricalBelief
from project.belief.composed_belief import ComposedBelief
from project.belief.normal_belief import NormalBelief
from project.prior.gaussian_prior import GaussianPrior
from project.prior.softmax_normal_prior import SoftmaxNormalPrior
from project.reconstructor.mock_reconstructor import MockReconstructor
from project.samplers.bayesian_sampler import BayesianSampler
from project.samplers.sampler import Sampler
from project.samplers.sde_sampler import SDESampler
from project.scheduler.ole_scheduler import OLEScheduler

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


def _test_sampler(SamplerClass: Any, **sampler_kwargs):
    shape_categorical = (5,)
    shape_continuous = (3, 4)
    batch_size = 4
    num_classes = 3
    prior = SoftmaxNormalPrior(shape=shape_categorical, num_classes=3, device=device)
    categorical_belief = CategoricalBelief(
        shape=shape_categorical,
        num_classes=3,
        categorical_prior=prior,
        device=str(device),
        key="c",
        scheduler=OLEScheduler(
            initial_precision=0.5,
            final_precision=50.0,
        ),
    )
    normal_prior = GaussianPrior(
        shape=shape_continuous, device=str(device), initial_precision=0.5
    )
    normal_belief = NormalBelief(
        normal_prior=normal_prior,
        shape=shape_continuous,
        device=str(device),
        key="x",
        initial_precision=0.5,
        final_precision=200000.0,
    )
    belief = ComposedBelief(
        beliefs=[categorical_belief, normal_belief],
        device=str(device),
    )
    samples = TensorDict(
        {
            "c": one_hot(
                torch.randint(
                    0, num_classes, (batch_size, *shape_categorical), device=device
                ),
                num_classes=num_classes,
            ),
            "x": torch.randn((batch_size, *shape_continuous), device=device),
        },
        batch_size=(batch_size),
    )

    reconstructor = MockReconstructor(samples)
    sampler: Sampler = SamplerClass(
        name="test_sampler",
        belief=belief,
        reconstructor=reconstructor,
        device=device,
        **sampler_kwargs,
    )
    generated_samples, _ = sampler.sample_batch(
        batch_size=batch_size,
    )
    assert assert_allclose_td(generated_samples, samples, atol=1e-2, rtol=1e-2)


def test_sde_sampler():
    _test_sampler(SDESampler, method="srk")


def test_bayesian_sampler():
    _test_sampler(BayesianSampler)
