from typing import Optional, Sequence, TypedDict

import numpy as np
import torch
from tensordict import TensorDict

from project.belief.belief import Belief
from project.class_conditioner.class_conditioner import ClassConditioner
from project.conditioner.conditioner import Conditioner
from project.reconstructor.reconstructor import SampleReconstructor
from project.samplers.sampler import Sampler


class RestartTimes(TypedDict):
    min_time: float
    max_time: float
    num_steps: int


class RestartSampler(Sampler):
    def __init__(
        self,
        name: str,
        belief: Belief,
        reconstructor: SampleReconstructor,
        class_conditioner: Optional[ClassConditioner] = None,
        device: torch.device = torch.device("cpu"),
        restart_times: list[float] | RestartTimes = {
            "min_time": 0.03,
            "max_time": 0.4,
            "num_steps": 10,
        },
        num_function_evaluations: int = 50,
        t_with_perfect_reconstruction: float = 0.6,
    ):
        super().__init__(name, belief, reconstructor, class_conditioner, device)
        if not isinstance(restart_times, Sequence):
            self.restart_times: list[float] = np.linspace(
                restart_times["min_time"],
                restart_times["max_time"],
                restart_times["num_steps"],
                endpoint=True,
            ).tolist()
        else:
            self.restart_times = restart_times
        # print(f"Restart times: {self.restart_times}")
        self.num_function_evaluations = num_function_evaluations
        self.t_with_perfect_reconstruction = t_with_perfect_reconstruction

    def sample_bayesian(
        self, belief: TensorDict, t_start: float, num_function_evaluations: int
    ) -> TensorDict:
        """Use a Bayesian sampler to integrate the belief to a sample at t=1."""
        assert t_start <= self.t_with_perfect_reconstruction, (
            "t_start must be less than t_with_perfect_reconstruction"
        )
        ts = np.linspace(
            t_start,
            self.t_with_perfect_reconstruction,
            num_function_evaluations,
            endpoint=True,
        )
        with torch.no_grad():
            for i, t in enumerate(ts[:-1]):
                belief["t"] = torch.full(belief.batch_size, t, device=self.device)
                reconstruction = self.reconstructor(belief)
                precision_increment = self.belief.get_precision(
                    torch.full(belief.batch_size, ts[i + 1], device=self.device)
                ) - self.belief.get_precision(
                    torch.full(belief.batch_size, t, device=self.device)
                )
                self.belief.sample_belief_update_inplace(
                    belief, reconstruction, precision_increment
                )
                del reconstruction
                # drift = self.belief.get_sde_drift(belief, reconstruction)
                # diffusion = self.belief.get_sde_diffusion(belief)
                # score = self.belief.get_score(belief, reconstruction)
                # derivative = drift - 0.5 * diffusion**2 * score  # type: ignore
                # mask = belief['mask']
                # del belief['mask']
                # del belief['t']
                # belief += derivative * (ts[i + 1] - t)
                # belief['mask'] = mask
            belief["t"] = torch.full(belief.batch_size, ts[-1], device=self.device)
            sample = self.reconstructor(belief)
            del belief
        return sample

    def sample_batch(
        self,
        batch_size: int,
        belief_timesteps_to_save: list[float] = [],
        conditioner: Conditioner | None = None,
    ) -> tuple[TensorDict, list[TensorDict]]:
        with torch.no_grad():
            belief = self.belief.sample_initial_belief(batch_size)
            initial_belief = belief.clone()
            classifications = None
            if self.class_conditioner is not None:
                classifications = self.class_conditioner.sample_classifications(
                    batch_size, device=self.device
                )
                self.class_conditioner.write_sample_classifications_to_belief(
                    classifications, belief
                )
            sample = None
            for t in self.restart_times:
                sample = self.sample_bayesian(belief, t, self.num_function_evaluations)
                del belief
                belief = initial_belief.clone()
                if classifications is not None:
                    self.class_conditioner.write_sample_classifications_to_belief(
                        classifications, belief
                    )
                precision = self.belief.get_precision(
                    torch.full(belief.batch_size, t, device=self.device)
                )
                belief["t"] = torch.full(belief.batch_size, t, device=belief.device)
                self.belief.sample_belief_update_inplace(belief, sample, precision)
        del initial_belief, belief, classifications
        assert sample is not None, "Sample should not be None"
        return sample, []
