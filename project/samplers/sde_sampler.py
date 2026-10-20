from typing import Optional

import torch
import torchsde
from torch import nn

from project.belief.belief import Belief
from project.class_conditioner.class_conditioner import ClassConditioner
from project.conditioner.conditioner import Conditioner
from project.reconstructor.reconstructor import SampleReconstructor
from project.samplers.sampler import Sampler
from project.utils.tensor_utils import expand_tensor, flatten_tensordict, unflatten_tensor


class BSI_SDE(nn.Module):
    def __init__(
        self,
        belief: Belief,
        reconstructor: SampleReconstructor,
        belief_shapes: dict[str, tuple[int, ...]],
        batch_size: int,
        class_conditioner: Optional[ClassConditioner] = None,
        device: torch.device = torch.device("cpu"),
        conditioner: Optional[Conditioner] = None,
    ):
        super().__init__()
        self.noise_type = "diagonal"
        self.sde_type = "ito"
        self.belief = belief
        self.reconstructor = reconstructor
        self.belief_shapes = belief_shapes
        self.class_conditioner = class_conditioner
        self.device = device
        self.classifications: Optional[torch.Tensor] = None
        if self.class_conditioner is not None:
            self.classifications = self.class_conditioner.sample_classifications(
                batch_size, device=self.device
            )
        self.conditioner = conditioner
        self.num_calls = 0

    def get_belief(self, t, y):
        belief = unflatten_tensor(y, shapes=self.belief_shapes, num_batchdims=1)
        batch_size = belief.batch_size
        belief["t"] = expand_tensor(t, batch_size)
        if self.class_conditioner is not None:
            if self.classifications is None:
                raise ValueError("Classifications must be set before calling f()")
            assert batch_size == self.classifications.shape[:1], (
                "Batch size mismatch between belief and classifications"
            )
            self.class_conditioner.write_sample_classifications_to_belief(
                self.classifications, belief
            )
        return belief

    # drift computation
    def f(self, t, y):
        belief = self.get_belief(t, y)
        reconstruction = self.reconstructor(belief)
        self.num_calls += 1
        if self.conditioner is not None:
            self.conditioner(reconstruction)
        drift = self.belief.get_sde_drift(belief, reconstruction)
        flattened = flatten_tensordict(drift)
        return flattened

    # diffusion computation
    def g(self, t, y):
        belief = self.get_belief(t, y)
        diffusion = self.belief.get_sde_diffusion(belief)
        return flatten_tensordict(diffusion)


class SDESampler(Sampler):
    def __init__(
        self,
        name: str,
        belief: Belief,
        reconstructor: SampleReconstructor,
        class_conditioner: Optional[ClassConditioner] = None,
        device: torch.device = torch.device("cpu"),
        method: str = "euler",
        atol: float = 5e-2,
        rtol: float = 5e-2,
        adaptive: bool = True,
        dt: float = 0.01,
    ):
        super().__init__(name, belief, reconstructor, class_conditioner, device)
        self.method = method
        self.atol = atol
        self.rtol = rtol
        self.adaptive = adaptive
        self.dt = dt

    def sample_batch(
        self,
        batch_size,
        belief_timesteps_to_save=[],
        conditioner=None,
    ):
        with torch.no_grad():
            initial_beliefs = self.belief.sample_initial_belief(batch_size)
            belief_shapes = {
                k: tuple(v.shape[1:]) for k, v in initial_beliefs.items() if k != "t"
            }
            bsi_sde = BSI_SDE(
                belief=self.belief,
                reconstructor=self.reconstructor,
                belief_shapes=belief_shapes,
                batch_size=batch_size,
                class_conditioner=self.class_conditioner,
                device=self.device,
                conditioner=conditioner,
            )

            y_0 = flatten_tensordict(initial_beliefs, ignore_keys={"t"})
            belief_timesteps_to_save = sorted(belief_timesteps_to_save)
            zero_added = False
            if 0 not in belief_timesteps_to_save:
                # If the first timestep to save is not 0, we add it
                belief_timesteps_to_save = [0.0] + belief_timesteps_to_save
                zero_added = True
            one_added = False
            if 1 not in belief_timesteps_to_save:
                belief_timesteps_to_save.append(1.0)
                one_added = True
            belief_series_tensor = torchsde.sdeint(
                bsi_sde,
                y_0,
                belief_timesteps_to_save,
                method=self.method,
                adaptive=self.adaptive,
                dt=self.dt,
                atol=self.atol,
                rtol=self.rtol,
            )
            # print("Number of calls to reconstructor:", bsi_sde.num_calls)
            belief_series = [
                unflatten_tensor(
                    belief_series_tensor[i], shapes=belief_shapes, num_batchdims=1
                )
                for i in range(len(belief_timesteps_to_save))
            ]
            for t, belief in zip(belief_timesteps_to_save, belief_series):
                belief["t"] = torch.full((batch_size,), t, device=self.device)
                if (
                    self.class_conditioner is not None
                    and bsi_sde.classifications is not None
                ):
                    self.class_conditioner.write_sample_classifications_to_belief(
                        bsi_sde.classifications, belief
                    )
            final_sample = self.belief.sample_from_belief(belief_series[-1])
            if self.class_conditioner is not None and bsi_sde.classifications is not None:
                self.class_conditioner.write_sample_classifications_to_belief(
                    bsi_sde.classifications, final_sample
                )
            if zero_added:
                # If we added the initial timestep, we need to remove it from the belief series
                belief_series = belief_series[1:]
            if one_added:
                # If we added the final timestep, we need to remove it from the belief series
                belief_series = belief_series[:-1]
            return final_sample, belief_series

    def to(self, device: torch.device) -> None:
        self.belief.to(device)
        self.reconstructor.to(device)
        self.device = device
