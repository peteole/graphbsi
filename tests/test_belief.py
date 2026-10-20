import random

import torch
from tensordict import (  # type: ignore
    TensorDict,
    assert_allclose_td,
)

from project.belief.belief import Belief
from project.belief.categorical_belief import CategoricalBelief
from project.belief.composed_belief import ComposedBelief
from project.belief.normal_belief import NormalBelief
from project.prior.gaussian_prior import GaussianPrior
from project.prior.softmax_normal_prior import SoftmaxNormalPrior
from project.scheduler.ole_scheduler import OLEScheduler

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


def assert_moments_match(
    ground_truth_samples: torch.Tensor,
    predicted_samples: torch.Tensor,
    z: float = 5.0,
):
    """Compare mean and std of two independent sample sets within z standard errors."""
    x = ground_truth_samples.float()
    y = predicted_samples.float()
    n = x.shape[0]
    mean_x, mean_y = x.mean(dim=0), y.mean(dim=0)
    std_x, std_y = x.std(dim=0), y.std(dim=0)
    mean_tol = z * torch.sqrt((std_x**2 + std_y**2) / n) + 1e-6
    std_tol = z * torch.sqrt((std_x**2 + std_y**2) / (2 * n)) + 1e-6
    assert torch.all((mean_x - mean_y).abs() <= mean_tol), (
        f"Means do not match: {mean_x} != {mean_y}"
    )
    assert torch.all((std_x - std_y).abs() <= std_tol), (
        f"Stds do not match: {std_x} != {std_y}"
    )


def assert_moments_match_tensordict(
    ground_truth_samples: TensorDict, predicted_samples: TensorDict
):
    for key in ground_truth_samples.keys():  # type: ignore
        assert_moments_match(ground_truth_samples[key], predicted_samples[key])


def _test_precision_additivity(
    ground_truth: TensorDict,
    belief: Belief,
    precision1: TensorDict,
    precision2: TensorDict,
    num_samples: int = 1000,
):
    # The inputs have batch size (1,); broadcast them to num_samples draws.
    ground_truth = ground_truth.expand(num_samples)
    initial_belief = belief.sample_initial_belief(ground_truth.batch_size)
    precision1 = precision1.expand(num_samples)
    precision2 = precision2.expand(num_samples)
    separately_updated_belief = initial_belief.clone()
    single_step_update_belief = initial_belief.clone()
    update_precision_0_belief = initial_belief.clone()
    belief.sample_belief_update_inplace(
        separately_updated_belief, ground_truth, precision1
    )
    belief.sample_belief_update_inplace(
        separately_updated_belief, ground_truth, precision2
    )
    belief.sample_belief_update_inplace(
        single_step_update_belief, ground_truth, precision1 + precision2
    )
    belief.sample_belief_update_inplace(
        update_precision_0_belief,
        ground_truth,
        precision1 * 0,  # type: ignore
    )
    assert_moments_match_tensordict(
        separately_updated_belief,
        single_step_update_belief,
    )
    assert assert_allclose_td(
        update_precision_0_belief,
        initial_belief,
    ), f"Beliefs do not match: {update_precision_0_belief} != {initial_belief}"


def _test_precision_schedule(belief: Belief):
    # Test precision at t=0 is 0
    precision = belief.get_precision(torch.zeros(1, device=device))
    assert_allclose_td(precision, 0 * precision)  # type: ignore

    # test that precision is strictly increasing
    t = torch.linspace(0, 1, 1000, device=device)
    precisions = belief.get_precision(t)
    for key, precision in precisions.items():
        assert torch.all(precision[1:] >= precision[:-1]), (
            f"Precision is not monotonically increasing for {key} at t={t}"
        )

    # Test the derivative
    dt = 1e-4
    dprecision_dt = belief.get_dprecision_dt(t)
    dprecision_dt_approx = (
        belief.get_precision(t + dt) - belief.get_precision(t - dt)
    ) / (2 * dt)
    assert_allclose_td(dprecision_dt, dprecision_dt_approx, atol=1e-2, rtol=1e-2)


def test_categorical_belief():
    torch.manual_seed(0)
    random.seed(0)
    num_classes = 10
    shape = (2,)
    categorical_ground_truth = torch.randint(0, num_classes - 1, (1,) + shape).to(device)
    categorical_ground_truth = torch.nn.functional.one_hot(
        categorical_ground_truth, num_classes=num_classes
    ).float()
    ground_truth = TensorDict({"categorical": categorical_ground_truth}, batch_size=(1,))
    precision1 = TensorDict(
        {"categorical": torch.ones((1,)).to(device) * random.uniform(0, 10)},
        batch_size=(1,),
    )

    precision2 = TensorDict(
        {"categorical": torch.ones((1,)).to(device) * random.uniform(0, 10)},
        batch_size=(1,),
    )
    categorical_prior = SoftmaxNormalPrior(
        shape=shape,
        num_classes=num_classes,
        device=device,
        initial_belief_std=0.1,
    )
    belief = CategoricalBelief(
        categorical_prior=categorical_prior,
        shape=shape,
        num_classes=num_classes,
        device=device,
        key="categorical",
        scheduler=OLEScheduler(initial_precision=1.0, final_precision=1e3),
    )
    _test_precision_additivity(
        ground_truth=ground_truth,
        belief=belief,
        precision1=precision1,
        precision2=precision2,
        num_samples=200000,
    )
    _test_precision_schedule(belief)


def test_normal_belief():
    torch.manual_seed(0)
    random.seed(0)
    shape = (2,)
    normal_ground_truth = torch.randn((1,) + shape).to(device)
    ground_truth = TensorDict({"normal": normal_ground_truth}, batch_size=(1,))
    precision1 = TensorDict(
        {"normal": torch.ones((1,)).to(device) * random.uniform(0, 10)}, batch_size=(1,)
    )
    precision2 = TensorDict(
        {"normal": torch.ones((1,)).to(device) * random.uniform(0, 10)}, batch_size=(1,)
    )
    normal_prior = GaussianPrior(
        shape=shape,
        device=device,
        initial_precision=5.0,
    )
    belief = NormalBelief(
        normal_prior=normal_prior,
        shape=shape,
        device=device,
        key="normal",
        final_precision=1e3,
    )
    _test_precision_additivity(
        ground_truth=ground_truth,
        belief=belief,
        precision1=precision1,
        precision2=precision2,
        num_samples=100000,
    )
    _test_precision_schedule(belief)


def test_composed_belief():
    torch.manual_seed(0)
    random.seed(0)
    shape = (2, 2)
    num_classes = 5
    normal_ground_truth = torch.randn((1,) + shape).to(device)
    categorical_ground_truth = torch.randint(0, num_classes - 1, (1,) + shape).to(device)
    categorical_ground_truth = torch.nn.functional.one_hot(
        categorical_ground_truth, num_classes=num_classes
    ).float()
    ground_truth = TensorDict(
        {"normal": normal_ground_truth, "categorical": categorical_ground_truth},
        batch_size=(1,),
    )
    precision1 = TensorDict(
        {
            "normal": torch.ones((1,)).to(device) * random.uniform(0, 10),
            "categorical": torch.ones((1,)).to(device) * random.uniform(0, 10),
        },
        batch_size=(1,),
    )

    precision2 = TensorDict(
        {
            "normal": torch.ones((1,)).to(device) * random.uniform(0, 10),
            "categorical": torch.ones((1,)).to(device) * random.uniform(0, 10),
        },
        batch_size=(1,),
    )
    normal_prior = GaussianPrior(
        shape=shape,
        device=device,
        initial_precision=1.0,
    )
    normal_belief = NormalBelief(
        normal_prior=normal_prior,
        shape=shape,
        device=device,
        key="normal",
        final_precision=1e3,
    )

    categorical_prior = SoftmaxNormalPrior(
        shape=shape,
        num_classes=num_classes,
        device=device,
        initial_belief_std=0.1,
    )

    categorical_belief = CategoricalBelief(
        categorical_prior=categorical_prior,
        shape=shape,
        num_classes=num_classes,
        device=device,
        key="categorical",
        scheduler=OLEScheduler(initial_precision=1.0, final_precision=50.0),
    )

    belief = ComposedBelief(beliefs=[normal_belief, categorical_belief], device=device)

    _test_precision_additivity(
        ground_truth=ground_truth,
        belief=belief,
        precision1=precision1,
        precision2=precision2,
        num_samples=200000,
    )
    _test_precision_schedule(belief)
