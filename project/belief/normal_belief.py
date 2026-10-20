from math import log, prod

import torch
from tensordict import TensorDict  # type: ignore
from torch.distributions import Normal

from project.belief.belief import Belief
from project.prior.prior import Prior
from project.utils.tensor_utils import expand_tensor


class NormalBelief(Belief):
    @classmethod
    def from_config(
        cls,
        normal_prior: Prior,
        shape: tuple,
        device: str | torch.device = "cpu",
        key: str = "normal",
        final_precision: float = 1e3,
        initial_precision: float = 1.0,
    ):
        return cls(
            normal_prior=normal_prior,
            shape=shape,
            device=device,
            key=key,
            final_precision=final_precision,
            initial_precision=initial_precision,
        )

    def __init__(
        self,
        normal_prior: Prior,
        shape: tuple,
        device: str | torch.device = "cpu",
        key: str = "normal",
        final_precision: float = 1e3,
        initial_precision: float = 1.0,
    ):
        super().__init__(
            belief_shape={f"{key}.mean": shape, f"{key}.precision": (1,)},
            sample_shape={key: shape},
        )
        self.key = key
        self.shape = shape
        self.final_precision = final_precision
        self.normal_prior = normal_prior
        self.initial_precision = initial_precision
        self.device = torch.device(device)

    def sample_initial_belief(self, batch_size):
        if isinstance(batch_size, int):
            batch_size = (batch_size,)
        return TensorDict(
            {
                f"{self.key}.mean": self.normal_prior.sample(batch_size),
                f"{self.key}.precision": torch.full(
                    batch_size, self.initial_precision, device=self.device
                ),
            },
            batch_size=batch_size,
            device=self.device,
        )

    def sample_from_belief(self, belief):
        mean = belief[f"{self.key}.mean"]
        precision = belief[f"{self.key}.precision"]
        normal_belief = Normal(mean, expand_tensor(1 / torch.sqrt(precision), mean.shape))
        return TensorDict(
            {self.key: normal_belief.sample()}, batch_size=mean.shape, device=self.device
        )

    def sample_belief_update_inplace(self, belief, ground_truth, precision):
        sample_mean = ground_truth[self.key]
        sample_precision = expand_tensor(precision[self.key], sample_mean.shape)
        observed_sample = torch.normal(sample_mean, 1 / torch.sqrt(sample_precision))
        observed_sample[sample_precision == 0] = sample_mean[
            sample_precision == 0
        ]  # avoid NaN with 0 precision
        belief_mean = belief[f"{self.key}.mean"]
        belief_precision = expand_tensor(
            belief[f"{self.key}.precision"], sample_mean.shape
        )
        updated_belief_mean = (
            belief_precision * belief_mean + sample_precision * observed_sample
        ) / (belief_precision + sample_precision)

        # Equivalent: belief_mean += (sample_mean - belief_mean) * sample_precision / ( belief_precision + sample_precision) + sqrt(sample_precision) / (belief_precision + sample_precision) * N(0,I)
        updated_belief_precision = belief[f"{self.key}.precision"] + precision[self.key]
        belief[f"{self.key}.mean"] = updated_belief_mean
        belief[f"{self.key}.precision"] = updated_belief_precision

    def dKL_dt(self, belief, ground_truth, reconstruction, t):
        dprecision_dt = self.get_dprecision_dt(t)[self.key]
        return (
            0.5
            * ((ground_truth[self.key] - reconstruction[self.key]) ** 2).sum(
                dim=list(range(1, ground_truth[self.key].ndim))
            )
            * dprecision_dt
        )

    def get_precision(self, t):
        return TensorDict(
            {
                self.key: self.initial_precision
                * torch.exp(t * log(self.final_precision / self.initial_precision))
                - self.initial_precision
            },
            batch_size=t.shape,
        )

    def log_prob(self, belief, sample):
        mean = belief[f"{self.key}.mean"]
        precision = belief[f"{self.key}.precision"]
        std = 1 / torch.sqrt(expand_tensor(precision, mean.shape))
        return (
            Normal(
                mean,
                std,
            )
            .log_prob(sample[self.key])
            .sum(dim=list(range(1, sample[self.key].ndim)))
        )

    def to(self, device):
        super().to(device)
        self.normal_prior.to(device)
        self.device = device

    def get_sde_drift(self, belief, ground_truth) -> TensorDict:
        sample_mean = ground_truth[self.key]
        belief_mean = belief[f"{self.key}.mean"]
        belief_precision = expand_tensor(
            belief[f"{self.key}.precision"], sample_mean.shape
        )
        t = belief["t"]
        dprecision_dt = self.get_dprecision_dt(t)[self.key]
        expanded_dprecision_dt = expand_tensor(dprecision_dt, sample_mean.shape)
        return TensorDict(
            {
                f"{self.key}.mean": (
                    (sample_mean - belief_mean)
                    * expanded_dprecision_dt
                    / belief_precision
                ),
                f"{self.key}.precision": dprecision_dt,
            },
            batch_size=belief.batch_size,
            device=self.device,
        )

    def get_sde_diffusion(self, belief) -> TensorDict:
        t = belief["t"]
        dprecision_dt = self.get_dprecision_dt(t)[self.key]
        precision = belief[f"{self.key}.precision"]
        precision = expand_tensor(precision, belief[f"{self.key}.mean"].shape)
        expanded_dprecision_dt = expand_tensor(
            dprecision_dt, belief[f"{self.key}.mean"].shape
        )
        return TensorDict(
            {
                f"{self.key}.mean": expanded_dprecision_dt.sqrt() / precision,
                f"{self.key}.precision": torch.zeros_like(
                    dprecision_dt
                ),  # No diffusion for precision,
            },
            batch_size=belief.batch_size,
            device=self.device,
        )

    def get_dimentionality(self) -> int:
        return prod(self.shape) + 1  # mean + precision

    def get_score(self, belief: TensorDict, ground_truth: TensorDict) -> TensorDict:
        t = belief["t"]
        belief_mean = belief[f"{self.key}.mean"]
        belief_precision = belief[f"{self.key}.precision"]
        belief_precision = expand_tensor(belief_precision, belief_mean.shape)
        accumulated_observation_precision = self.get_precision(t)[self.key]
        accumulated_observation_precision = expand_tensor(
            accumulated_observation_precision, belief_mean.shape
        )
        return TensorDict(
            {
                f"{self.key}.mean": accumulated_observation_precision
                * ground_truth[self.key]
                - belief_precision * belief_mean,
                f"{self.key}.precision": torch.zeros(
                    belief.batch_size, device=belief.device
                ),  # No score diffusion, therefore no correction necessary
            },
            batch_size=belief.batch_size,
            device=self.device,
        )

    def get_flow(self, belief: TensorDict, ground_truth: TensorDict) -> TensorDict:
        sample = ground_truth[self.key]
        belief_mean = belief[f"{self.key}.mean"]
        t = belief["t"]
        dprecision_dt = self.get_dprecision_dt(t)[self.key]
        dprecision_dt = expand_tensor(dprecision_dt, sample.shape)
        precision = self.get_precision(t)[self.key]
        precision = expand_tensor(precision, sample.shape)
        initial_precision = torch.full_like(precision, self.initial_precision)
        return TensorDict(
            {
                f"{self.key}.mean": (
                    dprecision_dt
                    * (
                        (precision / 2 + initial_precision)
                        / (initial_precision + precision)**2
                        * sample
                        - belief_mean / 2 / (precision + initial_precision)
                    )
                ),
                f"{self.key}.precision": dprecision_dt,
            },
            batch_size=belief.batch_size,
            device=self.device,
        )
