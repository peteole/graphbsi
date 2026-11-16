"""Parity check for the additive sampler integration.

A: the new CorrectedVariableNoiseStep driven through the unmodified StepSampler
B: an inline reference loop with the correction formulas written out
C: CorrectedVariableNoiseStep with both weights 0 vs the released VariableNoiseStep

A == B proves the wrapper applies the corrections in the right place, including
forgetting the previous reconstruction between batches. C == 0 proves the new
step method defaults to the released sampler.
"""

import sys
from itertools import pairwise
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import Model, SamplerConfig, timesteps, _simplex  # noqa: E402

from project.samplers.step_methods.corrected_variable_noise_step import (  # noqa: E402
    CorrectedVariableNoiseStep,
)
from project.samplers.step_methods.variable_noise_step import VariableNoiseStep  # noqa: E402
from project.samplers.step_sampler import StepSampler  # noqa: E402

m = Model("moses", device="cpu")
cfg = SamplerConfig(nfe=6, method="euler-maruyama", gamma=20, rho=0.7,
                    bg=0.3, ms=0.5, ms_clip=False, t_max=1.0)
B = 2
NUM_BATCHES = 2
ts = timesteps(cfg)


def make_sampler(step_method):
    sampler = StepSampler(
        name="parity",
        step_method=step_method,
        belief=m.belief,
        reconstructor=m.net,
        class_conditioner=m.class_conditioner,
        device=torch.device("cpu"),
        rho=cfg.rho,
        t_max=cfg.t_max,
        max_function_evals=cfg.nfe,
    )
    assert torch.allclose(sampler.timesteps, ts), (sampler.timesteps, ts)
    return sampler


def run_sampler(step_method):
    torch.manual_seed(7)
    sampler = make_sampler(step_method)
    out = []
    for _ in range(NUM_BATCHES):
        sample, _ = sampler.sample_batch(B)
        out.append({k: sample[k].clone() for k in ("X", "E")})
    return out


def run_reference():
    """StepSampler.sample_batch with the corrections written out inline."""
    torch.manual_seed(7)
    out = []
    for _ in range(NUM_BATCHES):
        params = m.belief.sample_initial_belief(B)
        params["t"] = torch.zeros(B)
        cls = m.class_conditioner.sample_classifications(B, device="cpu")
        m.class_conditioner.write_sample_classifications_to_belief(cls, params)
        prior = params.clone()
        prev, prev_dt = None, None
        for t1, t2 in pairwise(ts):
            params["t"] = torch.full((B,), float(t1))
            rec = m.reconstruct(params)
            dt = float(t2 - t1)
            for k in ("X", "E"):
                rec[k] = _simplex(rec[k] + cfg.bg * (rec[k] - torch.softmax(params[k], -1)))
            current = {k: rec[k].clone() for k in ("X", "E")}
            if prev is not None:
                for k in ("X", "E"):
                    rec[k] = rec[k] + cfg.ms * dt / prev_dt * (rec[k] - prev[k])
            prev, prev_dt = current, dt
            m.belief.discretize_belief_update_inplace(
                params, rec, prior, (t2 - t1), cfg.gamma, False, method=cfg.method
            )
        final = m.reconstruct(params)
        m.class_conditioner.write_sample_classifications_to_belief(cls, final)
        out.append({k: final[k].clone() for k in ("X", "E")})
    return out


def corrected(**kw):
    return CorrectedVariableNoiseStep(
        noise_scale=cfg.gamma, condition_on_prior=False, method=cfg.method, **kw
    )


with torch.no_grad():
    a = run_sampler(corrected(extrapolation=cfg.ms, extrapolation_clip=cfg.ms_clip,
                              guidance=cfg.bg))
    b = run_reference()
    c = run_sampler(corrected())
    d = run_sampler(VariableNoiseStep(noise_scale=cfg.gamma, condition_on_prior=False,
                                      method=cfg.method))

for i, (x, y) in enumerate(zip(a, b)):
    for k in ("X", "E"):
        print(f"A vs B  batch {i} {k}: max |diff| = {(x[k] - y[k]).abs().max().item():.3e}")
for i, (x, y) in enumerate(zip(c, d)):
    for k in ("X", "E"):
        print(f"C vs released  batch {i} {k}: max |diff| = {(x[k] - y[k]).abs().max().item():.3e}")
# The corrections must actually change something, or the test above is vacuous.
for k in ("X", "E"):
    print(f"corrected vs released  batch 0 {k}: max |diff| = "
          f"{(a[0][k] - d[0][k]).abs().max().item():.3e}")
