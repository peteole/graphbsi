from math import prod

from typing import Literal
import torch
from tensordict import TensorDict  # type: ignore
from torch.distributions import (
    Categorical,
)
from torch.nn.functional import softmax

from project.belief.belief import Belief
from project.prior.prior import Prior
from project.prior.softmax_normal_prior import SoftmaxNormalPrior
from project.scheduler.ole_scheduler import OLEScheduler
from project.scheduler.scheduler import Scheduler
from project.utils.tensor_utils import expand_tensor


class CategoricalBelief(Belief):
    def __init__(
        self,
        shape: tuple,
        num_classes: int,
        categorical_prior: Prior,
        device: str | torch.device = "cpu",
        key: str = "categorical",
        scheduler: Scheduler = OLEScheduler(),
        loss_mode: Literal["kl", "cross_entropy"] = "kl",
    ):
        super().__init__(
            belief_shape={key: shape + (num_classes,)},
            sample_shape={key: shape + (num_classes,)},
        )
        self.key = key
        self.num_classes = num_classes
        self.categorical_prior = categorical_prior
        self.shape = shape
        self.scheduler = scheduler
        self.device = torch.device(device)
        self.loss_mode = loss_mode

    def sample_initial_belief(self, batch_size):
        categorical_beliefs = self.categorical_prior.sample(batch_size)
        # categorical_beliefs = categorical_beliefs.log()
        return TensorDict(
            {self.key: categorical_beliefs}, batch_size=batch_size, device=self.device
        )

    def sample_from_belief(self, belief):
        probabilities = belief[self.key]
        probabilities = softmax(probabilities, dim=-1)
        categorical_belief = Categorical(probs=probabilities)
        res = torch.nn.functional.one_hot(
            categorical_belief.sample(), num_classes=self.num_classes
        )
        return TensorDict(
            {self.key: res}, batch_size=belief.batch_size, device=self.device
        )

    def log_prob(self, belief, sample):
        probabilities = belief[self.key]
        probabilities = softmax(probabilities, dim=-1)
        categorical_belief = Categorical(probs=probabilities)
        if (
            sample[self.key].ndim == len(self.shape) + 2
        ):  # One dim for batch, one for class
            sample_indices = sample[self.key].argmax(dim=-1)
        elif (
            sample[self.key].ndim == len(self.shape) + 1
        ):  # One dim for batch, none for class since it is not one hot encoded
            sample_indices = sample[self.key].long()
        else:
            raise ValueError(
                f"Sample shape {sample[self.key].shape} is not compatible with belief shape {self.shape}."
            )
        return categorical_belief.log_prob(sample_indices).sum(
            dim=list(range(1, sample_indices.ndim))
        )

    def sample_belief_update_inplace(self, belief, ground_truth, precision):
        # assume the ground truth is alread one-hot encoded
        categorical_ground_truth = self._get_one_hot_encoded_ground_truth(ground_truth)
        assert torch.allclose(
            categorical_ground_truth.sum(dim=-1),
            torch.ones_like(categorical_ground_truth.sum(dim=-1)),
        ), "Ground truth is not one-hot encoded"
        categorical_belief: torch.Tensor = belief[self.key]

        if categorical_ground_truth.shape[-1] != self.num_classes:
            raise ValueError(
                f"Ground truth shape {categorical_ground_truth.shape} does not match the number of classes {self.num_classes}. Shape should be {('batch'), self.shape + (self.num_classes,)}. Did you forget to one-hot encode the ground truth?"
            )
        if categorical_ground_truth.shape[-1 - len(self.shape) : -1] != self.shape:
            raise ValueError(
                f"Ground truth shape {categorical_ground_truth.shape} does not match the shape of the belief {self.shape}."
            )
        if categorical_ground_truth.shape != categorical_belief.shape:
            raise ValueError(
                f"Ground truth shape {categorical_ground_truth.shape} does not match the belief shape {categorical_belief.shape}."
            )

        batch_size = ground_truth.batch_size
        categorical_precision = precision[self.key]
        categorical_precision = expand_tensor(categorical_precision, batch_size)
        if categorical_precision.shape != batch_size:
            raise ValueError(
                f"Precision shape {categorical_precision.shape} does not match the batch size {batch_size}."
            )

        # Compute standard deviation
        std = torch.sqrt(
            expand_tensor(categorical_precision, categorical_ground_truth.shape)
        )

        # Generate noisy measurement
        noisy_measurement = torch.normal(
            expand_tensor(categorical_precision, categorical_ground_truth.shape)
            * categorical_ground_truth,
            std=std,
        )

        belief[self.key] += noisy_measurement
        # Clean up intermediate tensors
        del (
            categorical_ground_truth,
            categorical_belief,
            categorical_precision,
            std,
            noisy_measurement,
        )

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
        if method == "euler-maruyama":
            super().discretize_belief_update_inplace(
                belief, ground_truth, prior, dt, noise_scale, condition_on_prior, method
            )
        elif method == "ornstein-uhlenbeck":
            # Implement the Ornstein-Uhlenbeck process discretization here
            categorical_ground_truth = self._get_one_hot_encoded_ground_truth(
                ground_truth
            )
            categorical_belief = belief[self.key]
            batch_size = ground_truth.batch_size
            t = belief["t"]
            start_precision = self.get_precision(t)
            end_precision = self.get_precision(t + dt)
            precision = (start_precision + end_precision) / 2
            categorical_precision = expand_tensor(
                precision[self.key], categorical_ground_truth.shape
            )
            precision_increment = end_precision - start_precision
            categorical_precision_increment = expand_tensor(
                precision_increment[self.key], categorical_ground_truth.shape
            )
            if not isinstance(noise_scale, torch.Tensor):
                noise_scale = torch.tensor(
                    noise_scale, device=categorical_ground_truth.device, dtype=torch.float32
                )
            noise_scale = expand_tensor(noise_scale, categorical_ground_truth.shape)
            if condition_on_prior:
                kappa_dt = (
                    categorical_precision_increment
                    * (noise_scale - 1)
                    / 2
                    / categorical_precision
                )
                m = (noise_scale + 1) / (
                    noise_scale - 1
                ) * categorical_precision * ground_truth[self.key] + prior[self.key]
                noise_std = (
                    noise_scale
                    * categorical_precision
                    / (noise_scale - 1)
                    * (1 - torch.exp(-2 * kappa_dt))
                ).sqrt()
                belief[self.key] = (
                    m
                    + (belief[self.key] - m) * torch.exp(-kappa_dt)
                    + noise_std
                    * torch.randn(
                        categorical_ground_truth.shape, device=belief[self.key].device
                    )
                )
            else:
                dprecision_dt = self.get_dprecision_dt(t + dt / 2)[self.key]
                assert isinstance(self.categorical_prior, SoftmaxNormalPrior), (
                    "Categorical prior is not a SoftmaxNormalPrior."
                )

                initial_precision = self.categorical_prior.distribution.variance
                initial_mean = self.categorical_prior.distribution.mean
                # add batch dimension to initial_precision
                initial_precision = initial_precision.unsqueeze(0).expand(
                    batch_size
                    + tuple([-1 for _ in range(categorical_ground_truth.ndim - 1)])
                )
                initial_mean = initial_mean.unsqueeze(0).expand(
                    batch_size
                    + tuple([-1 for _ in range(categorical_ground_truth.ndim - 1)])
                )
                dprecision_dt = expand_tensor(
                    dprecision_dt, categorical_ground_truth.shape
                )
                kappa = (
                    (noise_scale - 1)
                    * dprecision_dt
                    / 2
                    / (initial_precision + categorical_precision)
                )
                kappa_dt = (
                    (noise_scale - 1)
                    * categorical_precision_increment
                    / 2
                    / (initial_precision + categorical_precision)
                )
                m = (
                    initial_mean
                    + (
                        categorical_precision
                        + 2
                        * (initial_precision + categorical_precision)
                        / (noise_scale - 1)
                    )
                    * categorical_ground_truth
                )
                variance = noise_scale * dprecision_dt
                noise = torch.randn_like(categorical_ground_truth)
                belief[self.key] = (
                    m
                    + (categorical_belief - m) * torch.exp(-kappa_dt)
                    + torch.sqrt(variance / 2 / kappa * (1 - torch.exp(-2 * kappa_dt)))
                    * noise
                )
        else:
            raise ValueError(
                f"Unknown method {method}. Supported methods are 'euler-maruyama' and 'ornstein-uhlenbeck'."
            )

    def get_precision(self, t):
        return TensorDict({self.key: self.scheduler.sample(t)}, batch_size=t.shape, device=self.device)

    def dKL_dt(self, belief, ground_truth, reconstruction, t):
        categorical_ground_truth = self._get_one_hot_encoded_ground_truth(ground_truth)
        categorical_reconstruction: torch.Tensor = reconstruction[self.key]  # type: ignore
        precision_factor = self.get_dprecision_dt(t)[self.key]
        if self.loss_mode == "kl":
            return (
                torch.sum(
                    (categorical_ground_truth - categorical_reconstruction) ** 2 / 2,
                    dim=list(range(1, categorical_ground_truth.ndim)),
                )
                * precision_factor
            )
        elif self.loss_mode == "cross_entropy":
            return (
                torch.sum(
                    -categorical_ground_truth
                    * torch.log(categorical_reconstruction + 1e-10),
                    dim=list(range(1, categorical_ground_truth.ndim)),
                )
                * precision_factor
            )
        else:
            raise ValueError(
                f"Unknown loss mode {self.loss_mode}. Supported modes are 'kl' and 'cross_entropy'."
            )

    def to(self, device):
        super().to(device)
        self.categorical_prior.to(device)
        self.device = device

    def get_sde_drift(self, belief, ground_truth) -> TensorDict:
        t = belief["t"]
        dprecision_dt = self.get_dprecision_dt(t)[self.key]
        categorical_ground_truth = self._get_one_hot_encoded_ground_truth(ground_truth)
        dprecision_dt = expand_tensor(dprecision_dt, categorical_ground_truth.shape)
        return TensorDict(
            {self.key: categorical_ground_truth * dprecision_dt},
            batch_size=belief.batch_size,
            device=self.device,
        )

    def get_sde_diffusion(self, belief) -> TensorDict:
        t = belief["t"]
        dprecision_dt = self.get_dprecision_dt(t)[self.key]
        dprecision_dt = expand_tensor(dprecision_dt, belief[self.key].shape)
        return TensorDict(
            {self.key: dprecision_dt.sqrt()},
            batch_size=belief.batch_size,
            device=self.device,
        )

    def get_score(self, belief: TensorDict, ground_truth: TensorDict) -> TensorDict:
        if not isinstance(self.categorical_prior, SoftmaxNormalPrior):
            raise ValueError(
                f"Score function is only implemented for SoftmaxNormalPrior but got {self.categorical_prior.__class__.__name__}."
                "Please implement the score function for your prior."
            )
        cat_belief: torch.Tensor = belief[self.key]
        cat_ground_truth = self._get_one_hot_encoded_ground_truth(ground_truth)
        t = belief["t"]
        precision = self.get_precision(t)[self.key]
        precision = expand_tensor(precision, cat_belief.shape)
        prior_mean = self.categorical_prior.distribution.mean
        # add batch dimension at the front
        prior_mean = prior_mean.unsqueeze(0).expand(
            belief.batch_size + tuple([-1 for _ in range(cat_belief.ndim - 1)])
        )
        prior_mean = expand_tensor(prior_mean, cat_belief.shape)
        base_precision = self.categorical_prior.distribution.variance.unsqueeze(0).expand(
            belief.batch_size + tuple([-1 for _ in range(cat_belief.ndim - 1)])
        )
        cat_score = (prior_mean + cat_ground_truth * precision - cat_belief) / (
            precision + base_precision
        )
        del cat_ground_truth, cat_belief, prior_mean, base_precision, precision, t
        return TensorDict(
            {self.key: cat_score},
            batch_size=belief.batch_size,
            device=self.device,
        )

    def get_prior_conditioned_score(
        self, belief: TensorDict, ground_truth: TensorDict, prior: TensorDict
    ) -> TensorDict:
        categorical_ground_truth = self._get_one_hot_encoded_ground_truth(ground_truth)
        categorical_prior: torch.Tensor = prior[self.key]
        categorical_belief: torch.Tensor = belief[self.key]
        t = belief["t"]
        categorical_precision = self.get_precision(t)[self.key]
        # add epsilon to avoid division by zero
        categorical_precision += 1e-8
        categorical_precision = expand_tensor(
            categorical_precision, categorical_belief.shape
        )

        return TensorDict(
            {
                self.key: categorical_ground_truth
                + (categorical_prior - categorical_belief) / categorical_precision
            },
            batch_size=belief.batch_size,
            device=self.device,
        )

    def get_flow(self, belief: TensorDict, ground_truth: TensorDict) -> TensorDict:
        if not isinstance(self.categorical_prior, SoftmaxNormalPrior):
            raise ValueError(
                f"Flow function is only implemented for SoftmaxNormalPrior but got {self.categorical_prior.__class__.__name__}."
                "Please implement the flow function for your prior."
            )
        d_precision_dt = self.get_dprecision_dt(belief["t"])[self.key]
        precision = self.get_precision(belief["t"])[self.key]
        cat_belief = belief[self.key]
        cat_ground_truth = self._get_one_hot_encoded_ground_truth(ground_truth)
        d_precision_dt = expand_tensor(d_precision_dt, cat_ground_truth.shape)
        precision = expand_tensor(precision, cat_belief.shape)

        base_precision = self.categorical_prior.distribution.variance.unsqueeze(0).expand(
            belief.batch_size + tuple([-1 for _ in range(cat_belief.ndim - 1)])
        )

        prior_mean = self.categorical_prior.distribution.mean
        prior_mean = prior_mean.unsqueeze(0).expand(
            belief.batch_size + tuple([-1 for _ in range(cat_belief.ndim - 1)])
        )
        prior_mean = expand_tensor(prior_mean, cat_belief.shape)
        return TensorDict(
            {
                self.key: d_precision_dt
                * (
                    (cat_belief - prior_mean) / 2 / (base_precision + precision)
                    + (base_precision + precision / 2)
                    / (base_precision + precision)
                    * cat_ground_truth
                )
            },
            batch_size=belief.batch_size,
            device=self.device,
        )

    def _get_one_hot_encoded_ground_truth(self, ground_truth: TensorDict) -> torch.Tensor:
        """
        Convert the ground truth to one-hot encoded format if it is not already.
        """
        ground_truth_tensor: torch.Tensor = ground_truth[self.key]
        if ground_truth_tensor.ndim == len(self.shape) + 2:
            return ground_truth_tensor
        elif (
            ground_truth_tensor.ndim == len(self.shape) + 1
            and not ground_truth_tensor.is_floating_point()
        ):
            return torch.nn.functional.one_hot(
                ground_truth_tensor.long(), num_classes=self.num_classes
            ).float()
        else:
            raise ValueError(
                f"Ground truth shape {ground_truth_tensor.shape} is not compatible with belief shape {self.shape}."
            )

    def expand_sample(self, sample: TensorDict) -> TensorDict:
        """
        Expand the sample to one-hot encoded format if it is not already.
        """
        expanded_sample = sample.clone()
        expanded_sample[self.key] = self._get_one_hot_encoded_ground_truth(
            expanded_sample
        )
        return expanded_sample

    def get_dimentionality(self) -> int:
        return self.num_classes * prod(self.shape)
