from typing import Optional

import numpy as np
import torch
import torchode as to
from tensordict import TensorDict

from project.belief.belief import Belief
from project.class_conditioner.class_conditioner import ClassConditioner
from project.conditioner.conditioner import Conditioner
from project.reconstructor.reconstructor import SampleReconstructor
from project.samplers.sampler import Sampler
from project.utils.tensor_utils import expand_tensor, flatten_tensordict, unflatten_tensor


class FlowMatchingSampler(Sampler):
    def __init__(
        self,
        name: str,
        belief: Belief,
        reconstructor: SampleReconstructor,
        class_conditioner: Optional[ClassConditioner] = None,  # noqa: F821
        device: torch.device = torch.device("cpu"),
        time_discretization_steps: int = 50,
    ):
        super().__init__(name, belief, reconstructor, class_conditioner, device)
        self.time_discretization_steps = time_discretization_steps

    def sample_batch(
        self,
        batch_size: int,
        belief_timesteps_to_save: list[float] = [],
        conditioner: Optional[Conditioner] = None,
    ):
        initial_beliefs = self.belief.sample_initial_belief(batch_size)
        classifications = (
            self.class_conditioner.sample_classifications(batch_size, device=self.device)
            if self.class_conditioner
            else None
        )
        belief_shapes = {
            k: tuple(v.shape[1:]) for k, v in initial_beliefs.items() if k != "t"
        }

        def get_belief(t, y):
            belief = unflatten_tensor(y, shapes=belief_shapes, num_batchdims=1)
            batch_size = belief.batch_size
            belief["t"] = expand_tensor(t, batch_size)
            if self.class_conditioner is not None:
                if classifications is None:
                    raise ValueError("Classifications must be set before calling f()")
                assert batch_size == classifications.shape[:1], (
                    "Batch size mismatch between belief and classifications"
                )
                self.class_conditioner.write_sample_classifications_to_belief(
                    classifications, belief
                )
            return belief

        def f(t, y):
            belief = get_belief(t, y)
            reconstruction = self.reconstructor(belief)
            result = self.belief.get_flow(belief,ground_truth=reconstruction)
            return flatten_tensordict(result)

        y_0 = flatten_tensordict(initial_beliefs, ignore_keys={"t"})
        term = to.ODETerm(f)
        step_method = to.Heun(term=term)
        step_size_controller = to.FixedStepController()
        solver = to.AutoDiffAdjoint(step_method, step_size_controller)
        belief_timesteps_to_save_extended = belief_timesteps_to_save.copy()
        dtype = torch.float32
        if len(belief_timesteps_to_save) == 0 or np.max(belief_timesteps_to_save) < 1.0:
            belief_timesteps_to_save_extended.append(1.0)
        t_eval = (
            torch.tensor(belief_timesteps_to_save_extended, device=self.device, dtype=dtype).repeat(
                (batch_size, 1)
            )
            if belief_timesteps_to_save_extended
            else None
        )
        sol = solver.solve(
            to.InitialValueProblem(
                y0=y_0,
                t_start=torch.zeros(batch_size, device=self.device, dtype=dtype),
                t_end=torch.ones(batch_size, device=self.device, dtype=dtype),
                t_eval=t_eval,
            ),
            dt0=torch.full((batch_size,), 1.0 / self.time_discretization_steps, device=self.device, dtype=dtype),
        )
        mean_stats = {k: torch.mean(v.float()).item() for k, v in sol.stats.items()}
        print(mean_stats)
        intermediate_beliefs: list[TensorDict] = []
        for i,t in enumerate(belief_timesteps_to_save):
            intermediate_beliefs.append(get_belief(sol.ts[:, i], sol.ys[:, i, :]))
        final_beliefs = get_belief(sol.ts[:, -1], sol.ys[:, -1, :])
        final_sample = self.reconstructor(final_beliefs)
        if self.class_conditioner is not None and classifications is not None:
            self.class_conditioner.write_sample_classifications_to_belief(
                classifications, final_sample
            )
        del final_beliefs
        return final_sample, intermediate_beliefs
