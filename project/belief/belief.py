from abc import abstractmethod

import torch
from tensordict import TensorDict  # type: ignore
from torch import Tensor

from project.utils.tensor_utils import expand_tensor


class Belief:
    def __init__(self, belief_shape: dict[str, tuple], sample_shape: dict[str, tuple]):
        self.belief_shape = belief_shape
        self.sample_shape = sample_shape

    @abstractmethod
    def sample_from_belief(self, belief: TensorDict) -> TensorDict:
        """
        Sample from the belief distribution.
        """

    @abstractmethod
    def log_prob(self, belief: TensorDict, sample: TensorDict) -> Tensor:
        """
        Compute the log probability of a sample given a belief.
        Args:
            belief (TensorDict): A batch of beliefs to sample from. Only uses the keys of this component.
            sample (TensorDict): A batch of samples to measure. Only uses the keys of this component.
        Returns:
            Tensor: A batch of log probabilities of the samples given the beliefs. Shape (*batch_size,).
        """

    @abstractmethod
    def sample_belief_update_inplace(
        self, belief: TensorDict, ground_truth: TensorDict, precision: TensorDict
    ) -> None:
        """
        Sample an update of a belief using a noisy measurement of the ground truth `ground_truth` with a precision `precision`.
        The update is done in place, so the belief is modified directly on the keys of this component. Repeatedly calling this method with the same ground truth should be equivalent to calling it once with the sum of all the precisions.
        Args:
            belief (TensorDict): A batch of beliefs to update. Only uses the keys of this component.
            ground_truth (TensorDict): A batch of ground truth samples to measure. Only uses the keys of this component.
            precision (TensorDict): The precision of the measurement. Same batch size as `ground_truth`. Precisions of 0 should not modify the belief while large precisions should move the belief towards the ground truth.
        """

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
        """
        Discretize a belief update step `dt` in-place. By default uses the euler-maruyama method, but subclasses may implement other discretization schemes.
        The update is done in place, so the belief is modified directly on the keys of this component.
        Args:
            belief (TensorDict): A batch of beliefs to update. Only uses the keys of this component.
            ground_truth (TensorDict): A batch of ground truth samples to measure. Only uses the keys of this component.
            dt (torch.Tensor): The time step for the discretization. Same batch size as `ground_truth`.
            noise_scale (float | torch.Tensor | None): Optional scaling factor for the noise. If None, it is set to 1 for the vanilla Bayesian update.
        """
        if method is None or method == "euler-maruyama":
            drift = self.get_sde_drift(belief, ground_truth)
            diffusion = self.get_sde_diffusion(belief)
            score = (
                self.get_score(belief, ground_truth)
                if not condition_on_prior
                else self.get_prior_conditioned_score(belief, ground_truth, prior)
            )
            for key in drift.keys():
                wiener_noise = torch.randn_like(drift[key]) * torch.sqrt(dt)
                if not isinstance(noise_scale, torch.Tensor):
                    noise_scale = torch.tensor(noise_scale, device=drift[key].device)
                expanded_noise_scale = expand_tensor(noise_scale, drift[key].shape)
                belief[key] += (
                    drift[key]
                    + (expanded_noise_scale - 1) / 2 * diffusion[key] ** 2 * score[key]
                ) * dt + torch.sqrt(expanded_noise_scale) * diffusion[key] * wiener_noise
                del wiener_noise, expanded_noise_scale
            del drift, diffusion, score, noise_scale
        else:
            raise NotImplementedError(
                f"Discretization method {method} is not implemented."
            )

    @abstractmethod
    def get_precision(self, t: Tensor) -> TensorDict:
        """
        Get the precision of the belief at time t. Must be differentiable with respect to t.
        Args:
            t (Tensor): The time at which to get the precision. Shape (batch_size,). All values are in [0, 1].

        Returns:
            TensorDict: The precision of the belief at time t. This can be different tensor for subcomponents of the belief. Batch size is the same as the shape of t.
        """

    def get_dprecision_dt(self, t: Tensor) -> TensorDict:
        """
        Get the derivative of the precision of the belief at time t.
        Args:
            t (Tensor): The time at which to get the precision derivative. Shape (batch_size,). All values are in [0, 1].

        Returns:
            TensorDict: The derivative of the precision of the belief at time t. This can be different tensor for subcomponents of the belief. Batch size is the same as the shape of t.
        """
        derivatives: dict[str, Tensor] = {}
        with torch.inference_mode(False):
            t_grad = t.clone().detach().requires_grad_(True)
            precisions = self.get_precision(t_grad)

            for key, precision in precisions.items():
                assert precision.shape == t_grad.shape, (
                    f"Precision shape {precision.shape} does not match the shape of t {t.shape}."
                )
                # Get derivative through gradient of sum
                grad = torch.autograd.grad(
                    precision.sum(), t_grad, create_graph=False, retain_graph=True
                )[0]
                # Detach the gradient to prevent memory leaks
                derivatives[key] = grad.detach()

            # Clean up
            del precisions, t_grad
            torch.cuda.empty_cache()

        return TensorDict(derivatives, batch_size=t.shape)

    @abstractmethod
    def sample_initial_belief(self, batch_size: int | tuple | torch.Size) -> TensorDict:
        """
        Sample an initial belief.
        Args:
            batch_size (int | tuple | torch.Size): The batch size of the initial belief.

        Returns:
            TensorDict: A batch of initial beliefs, only with the keys of this component.
        """

    @abstractmethod
    def dKL_dt(
        self,
        belief: TensorDict,
        ground_truth: TensorDict,
        reconstruction: TensorDict,
        t: Tensor,
    ) -> Tensor:
        """
        Compute the linear expansion of the KL divergence with respect to the time increment dt at time t.

        $
            lim_(dt -> 0) 1/dt * KL(p(next_belief|blief, ground_truth, precision(t+dt)-precision(t)) || p(next_belief|reconstruction, precision(t+dt)-precision(t))
            = dKL/dprecision * get_dprecision_dt(t)
        $

        Args:
            belief (TensorDict): A batch of beliefs to update. Only uses the keys of this component.
            ground_truth (TensorDict): A batch of ground truth samples to measure. Only uses the keys of this component.
            reconstruction (TensorDict): A batch of reconstructed samples to measure. Only uses the keys of this component.
            t (Tensor): The time at which to get the KL divergence derivative. Shape (batch_size,). All values are in [0, 1].

        Returns:
            Tensor: A batch of KL divergence derivatives with respect to t. Shape (batch_size,).
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(belief:{self.belief_shape}, sample: {self.sample_shape})"

    def to(self, device: torch.device) -> None:
        """
        Move the belief to a device.
        Args:
            device (torch.device): The device to move the belief to.
        """
        pass

    @abstractmethod
    def get_sde_drift(self, belief, ground_truth) -> TensorDict:
        """
        Get the SDE drift term for the belief update.
        Args:
            belief (TensorDict): The current belief state.
            ground_truth (TensorDict): The ground truth state to compare against.

        Returns:
            TensorDict: The drift term for the SDE update, for each component of the belief.
        """
        pass

    @abstractmethod
    def get_sde_diffusion(self, belief) -> TensorDict:
        """
        Get the SDE diffusion term for the belief update.
        Args:
            belief (TensorDict): The current belief state.

        Returns:
            TensorDict: The diffusion term for the SDE update, for each component of the belief.
        """
        pass

    @abstractmethod
    def get_score(self, belief: TensorDict, ground_truth: TensorDict) -> TensorDict:
        """
        Get the score function for the belief update. The score function is the gradient of the log probability of the belief given the ground truth:

        score(belief, ground_truth) = nabla_belief log p(belief | ground_truth)
        Args:
            belief (TensorDict): The current belief state.
            ground_truth (TensorDict): The ground truth state to compare against.

        Returns:
            TensorDict: The score function for the belief update, for each component of the belief.
        """
        pass

    @abstractmethod
    def get_prior_conditioned_score(
        self, belief: TensorDict, ground_truth: TensorDict, prior: TensorDict
    ) -> TensorDict:
        """
        Get the score function conditioned on the prior for the belief update. The score function is the gradient of the log probability of the belief given the ground truth:

        score(belief, ground_truth, prior) = nabla_belief log p(belief | ground_truth, belief["t"], prior)
        Args:
            belief (TensorDict): The current belief state.
            ground_truth (TensorDict): The ground truth state to compare against.
            prior (TensorDict): The prior state to condition on.

        Returns:
            TensorDict: The score function for the belief update, for each component of the belief.
        """
        pass

    @abstractmethod
    def get_flow(self, belief: TensorDict, ground_truth: TensorDict) -> TensorDict:
        """
        Get the conditional flow at the current belief with the ground truth for flow matching.
        Args:
            belief (TensorDict): The current belief state.
            ground_truth (TensorDict): The ground truth state to compare against.

        Returns:
            TensorDict: The flow term for the belief update, for each component of the belief.
        """
        pass

    def sample_belief_at_t(self, ground_truth: TensorDict, t: Tensor) -> TensorDict:
        """
        Sample a belief at a specific time t.
        Args:
            batch_size (int | tuple | torch.Size): The batch size of the belief.
            t (Tensor): The time at which to sample the belief. Shape (batch_size,).

        Returns:
            TensorDict: A batch of beliefs at time t, only with the keys of this component.
        """
        precision = self.get_precision(t)
        batch_size = ground_truth.batch_size
        belief = self.sample_initial_belief(batch_size)
        self.sample_belief_update_inplace(belief, ground_truth, precision)
        return belief

    @abstractmethod
    def get_dimentionality(self) -> int:
        """
        Get the dimensionality of the belief.
        Returns:
            int: The dimensionality of the belief.
        """
        pass

    def expand_sample(self, sample: TensorDict) -> TensorDict:
        """
        Expand the sample, if necessary, for example performing a one-hot encoding or similar operation.
        Args:
            sample (TensorDict): The sample to expand.
        Returns:
            TensorDict: The expanded sample.
        """
        return sample
