from project.belief.categorical_belief import CategoricalBelief
from project.prior.softmax_normal_prior import SoftmaxNormalPrior
from project.scheduler.ole_scheduler import OLEScheduler
from tensordict import TensorDict  # type: ignore
import torch

from project.utils.tensor_utils import expand_tensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _assert_same_distribution(x: torch.Tensor, y: torch.Tensor, name: str, z: float = 5.0):
    """Compare mean and std of two independent sample sets within z standard errors."""
    n = x.shape[0]
    mean_x, mean_y = x.mean(dim=0), y.mean(dim=0)
    std_x, std_y = x.std(dim=0), y.std(dim=0)
    mean_tol = z * torch.sqrt((std_x**2 + std_y**2) / n)
    std_tol = z * torch.sqrt((std_x**2 + std_y**2) / (2 * n))
    assert torch.all((mean_x - mean_y).abs() <= mean_tol), (
        f"Mean {name}: {mean_x} vs {mean_y} (tolerance {mean_tol})"
    )
    assert torch.all((std_x - std_y).abs() <= std_tol), (
        f"Std {name}: {std_x} vs {std_y} (tolerance {std_tol})"
    )


def test_ou_discretization():
    torch.manual_seed(0)
    shape = (1,)
    num_classes = 2
    key = "categorical"
    belief = CategoricalBelief(
        shape=shape,
        num_classes=num_classes,
        categorical_prior=SoftmaxNormalPrior(
            shape=shape, num_classes=num_classes, device=device, initial_belief_std=1.0
        ),
        scheduler=OLEScheduler(initial_precision=1.0, final_precision=10.0),
        key=key,
        device=device,
    )
    batch_size = 100000
    belief_params = belief.sample_initial_belief(batch_size)
    belief_params["t"] = torch.zeros(batch_size, device=device)
    belief_params_euler_update = belief_params.clone()
    belief_params_ornstein_uhlbeck_update = belief_params.clone()
    belief_params_bayesian_update = belief_params.clone()
    ground_truth = TensorDict(
        {
            key: torch.nn.functional.one_hot(
                torch.full((batch_size,) + shape, 0, dtype=torch.long, device=device),
                num_classes=num_classes,
            ).float()
        },
        batch_size=(batch_size,),
        device=device,
    )
    discretization = 2000
    t_end = 0.5
    belief.sample_belief_update_inplace(
        belief_params_bayesian_update,
        ground_truth,
        belief.get_precision(torch.tensor(t_end, device=device)),
    )
    dt = torch.tensor(t_end / discretization, device=device)
    noise_level = 0.0
    for t in torch.linspace(0, t_end, discretization, device=device):
        belief_params_euler_update["t"] = expand_tensor(t, (batch_size,))
        belief_params_ornstein_uhlbeck_update["t"] = expand_tensor(t, (batch_size,))
        belief.discretize_belief_update_inplace(
            belief_params_euler_update,
            ground_truth,
            belief_params,
            dt,
            noise_scale=noise_level,
            condition_on_prior=False,
            method="euler-maruyama",
        )
        belief.discretize_belief_update_inplace(
            belief_params_ornstein_uhlbeck_update,
            ground_truth,
            belief_params,
            dt,
            noise_scale=noise_level,
            condition_on_prior=False,
            method="ornstein-uhlenbeck",
        )
    mean_euler = belief_params_euler_update[key].mean(dim=0)
    mean_ornstein = belief_params_ornstein_uhlbeck_update[key].mean(dim=0)
    std_euler = belief_params_euler_update[key].std(dim=0)
    std_ornstein = belief_params_ornstein_uhlbeck_update[key].std(dim=0)
    assert torch.allclose(mean_euler, mean_ornstein, atol=5e-3), (
        f"Mean Euler: {mean_euler}, Mean Ornstein: {mean_ornstein}"
    )
    assert torch.allclose(std_euler, std_ornstein, atol=1e-2), (
        f"Std Euler: {std_euler}, Std Ornstein: {std_ornstein}"
    )
    # Discretized and exact updates start from independent draws.
    _assert_same_distribution(
        belief_params_euler_update[key], belief_params_bayesian_update[key], "Euler vs Bayesian"
    )


def test_ou_discretization_condition_on_prior():
    torch.manual_seed(0)
    shape = (1,)
    num_classes = 2
    key = "categorical"
    belief = CategoricalBelief(
        shape=shape,
        num_classes=num_classes,
        categorical_prior=SoftmaxNormalPrior(
            shape=shape, num_classes=num_classes, device=device, initial_belief_std=1.0
        ),
        scheduler=OLEScheduler(initial_precision=1.0, final_precision=10.0),
        key=key,
        device=device,
    )
    batch_size = 10000
    prior = belief.sample_initial_belief(batch_size)
    t_start = 0.05

    ground_truth = TensorDict(
        {
            key: torch.nn.functional.one_hot(
                torch.full((batch_size,) + shape, 0, dtype=torch.long, device=device),
                num_classes=num_classes,
            ).float()
        },
        batch_size=(batch_size,),
    )
    belief_params = prior.clone()
    belief_params["t"] = torch.full((batch_size,), t_start, device=device)
    belief.sample_belief_update_inplace(
        belief_params, ground_truth, belief.get_precision(belief_params["t"])
    )
    belief_params_euler_update = belief_params.clone()
    belief_params_ornstein_uhlbeck_update = belief_params.clone()
    belief_params_bayesian_update = belief_params.clone()
    discretization = 2000
    t_end = 0.5
    belief.sample_belief_update_inplace(
        belief_params_bayesian_update,
        ground_truth,
        (
            belief.get_precision(torch.tensor(t_end, device=device))
            - belief.get_precision(torch.tensor(t_start, device=device))
        ),
    )
    dt = torch.tensor((t_end - t_start) / discretization, device=device)
    noise_level = 0.0
    for t in torch.linspace(t_start, t_end, discretization, device=device):
        belief_params_euler_update["t"] = expand_tensor(t, (batch_size,))
        belief_params_ornstein_uhlbeck_update["t"] = expand_tensor(t, (batch_size,))
        belief.discretize_belief_update_inplace(
            belief_params_euler_update,
            ground_truth,
            prior,
            dt,
            noise_scale=noise_level,
            condition_on_prior=True,
            method="euler-maruyama",
        )
        belief.discretize_belief_update_inplace(
            belief_params_ornstein_uhlbeck_update,
            ground_truth,
            prior,
            dt,
            noise_scale=noise_level,
            condition_on_prior=True,
            method="ornstein-uhlenbeck",
        )
    mean_euler = belief_params_euler_update[key].mean(dim=0)
    mean_ornstein = belief_params_ornstein_uhlbeck_update[key].mean(dim=0)
    std_euler = belief_params_euler_update[key].std(dim=0)
    std_ornstein = belief_params_ornstein_uhlbeck_update[key].std(dim=0)
    assert torch.allclose(mean_euler, mean_ornstein, atol=5e-3), (
        f"Mean Euler: {mean_euler}, Mean Ornstein: {mean_ornstein}"
    )
    assert torch.allclose(std_euler, std_ornstein, atol=1e-2), (
        f"Std Euler: {std_euler}, Std Ornstein: {std_ornstein}"
    )
    # Discretized and exact updates start from independent draws.
    _assert_same_distribution(
        belief_params_euler_update[key], belief_params_bayesian_update[key], "Euler vs Bayesian"
    )
