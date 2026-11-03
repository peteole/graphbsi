"""Research harness: load a molecular checkpoint once, sample SMILES with a
configurable sampler and score them with the paper's benchmark code.

Sampling (GPU) and scoring (CPU) are decoupled so that many sampler variants
can be swept per model load: `sample_smiles` returns SMILES, `score_smiles`
runs the same `molecule_benchmarks.Benchmarker` as the paper.
"""

from __future__ import annotations

import json
import math
import os
import pickle
import sys
import time
from dataclasses import asdict, dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import Optional

import torch
from tensordict import TensorDict

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "research"))
os.chdir(REPO)

DEFAULT_CKPT = {
    "moses": "final_checkpoints/moses/last.ckpt",
    "guacamol": "final_checkpoints/guacamol/last.ckpt",
    "qm9": "final_checkpoints/qm9/last.ckpt",
}


@dataclass
class SamplerConfig:
    """All knobs of the research sampler. Defaults reproduce the paper sampler."""

    nfe: int = 50
    method: str = "euler-maruyama"  # or "ornstein-uhlenbeck"
    gamma: float = 10.0  # noise scale
    # Noise scale may vary over time: gamma(t) = gamma * (gamma_end/gamma)^(t/t_max)
    gamma_end: Optional[float] = None
    rho: float = 1.0
    t_max: float = 1.2
    # Temperature applied to reconstructed probabilities fed into the belief update
    tau_x: float = 1.0
    tau_e: float = 1.0
    # Temperature applied only in the final decoding (argmax ignores it unless sampling)
    final_decode: str = "argmax"  # or "sample"
    # Apply temperature only for t >= tau_from (fraction of t_max)
    tau_from: float = 0.0
    # Feed the previous reconstruction back in (self-conditioned checkpoints only)
    selfcond: bool = True
    # Multistep extrapolation of the reconstruction (no extra NFE):
    # x_used = x_n + ms * (x_n - x_{n-1}) * h_n / h_{n-1}; ms=0.5 ~ Adams-Bashforth 2
    ms: float = 0.0
    # Project the extrapolated reconstruction back onto the simplex
    ms_clip: bool = True
    # Apply extrapolation only for t/t_max in [ms_lo, ms_hi]; per-key weight multipliers
    ms_lo: float = 0.0
    ms_hi: float = 1.0
    ms_x: float = 1.0
    ms_e: float = 1.0
    # 2 = linear extrapolation from the last two reconstructions, 3 = quadratic
    # extrapolation from the last three. Neither costs a function evaluation.
    ms_order: int = 2
    # Belief guidance: push the reconstruction away from the belief's own current
    # estimate, x_used = x + bg * (x - softmax(belief)). Also free of extra NFE.
    bg: float = 0.0
    # Apply the guidance only for t/t_max in [bg_lo, bg_hi]
    bg_lo: float = 0.0
    bg_hi: float = 1.0
    # Hybrid integrator: use `method` while t/t_max < switch_at, then `method2`
    # with noise level `gamma2`. Ornstein-Uhlenbeck is stable at the high noise
    # that helps diversity, Euler is more accurate at low noise.
    method2: Optional[str] = None
    gamma2: Optional[float] = None
    switch_at: float = 0.5
    # Time fed to the *final* reconstruction. "end" is the time the belief has
    # actually reached; "last_step" is what the released StepSampler passes (the
    # previous grid point), which is inconsistent with the belief.
    final_t: str = "end"
    batch_size: int = 1024
    seed: int = 0
    extra: dict = field(default_factory=dict)

    def key(self) -> str:
        d = asdict(self)
        d.pop("batch_size")
        return json.dumps(d, sort_keys=True)


class Model:
    def __init__(self, dataset: str, ckpt: Optional[str] = None, device: str = "cuda",
                 selfcond: bool = False, rrwp_sc: int = 8):
        from hydra import compose, initialize_config_dir
        from hydra.utils import instantiate

        from project.utils import set_seed

        ckpt = ckpt or DEFAULT_CKPT[dataset]
        with initialize_config_dir(config_dir=str(REPO / "config"), version_base=None):
            cfg = compose(
                "train",
                overrides=[f"experiment={dataset}/reproduce", f"load_checkpoint={ckpt}", "wandb.mode=disabled"],
            )
        self.cfg = cfg
        torch.set_float32_matmul_precision(cfg.matmul_precision)
        rng = set_seed(cfg)
        dm = instantiate(cfg.data, rng=rng)
        dm.prepare_data()
        dm.setup("fit")
        self.datamodule = dm
        reconstructor = instantiate(cfg.reconstructor, dm)
        belief = instantiate(cfg.belief, datamodule=dm)
        cc = instantiate(cfg.class_conditioner, datamodule=dm)
        task = instantiate(
            cfg.task, reconstructor=reconstructor, belief=belief, class_conditioner=cc, datamodule=dm
        )
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        self.selfcond = bool(state.get("selfcond", False)) or selfcond
        if state.get("selfcond", False):
            self._add_selfcond(task, dm, state.get("rrwp_sc", rrwp_sc))
        missing, unexpected = task.load_state_dict(state["state_dict"], strict=False)
        assert not unexpected, unexpected
        assert not [m for m in missing if "ema" not in m], missing
        if self.selfcond and not state.get("selfcond", False):
            self._add_selfcond(task, dm, rrwp_sc)
        self.rrwp_sc = state.get("rrwp_sc", rrwp_sc)
        self.device = torch.device(device)
        task.to(self.device)
        task.eval()
        self.task = task
        self.belief = belief
        self.class_conditioner = cc
        self.net = task.ema_reconstructor if task.ema_reconstructor is not None else reconstructor
        self.net.eval()
        self.dataset = dataset
        self.ckpt = ckpt

    @staticmethod
    def _add_selfcond(task, dm, rrwp_sc: int) -> None:
        from selfcond import add_self_conditioning

        nx, ne = dm.get_num_node_features(), dm.get_num_edge_features()
        add_self_conditioning(task.reconstructor, nx, ne, rrwp_sc)
        if task.ema_reconstructor is not None:
            add_self_conditioning(task.ema_reconstructor.ema_model, nx, ne, rrwp_sc)
            for p in task.ema_reconstructor.ema_model.parameters():
                p.requires_grad_(False)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.task.reconstructor.parameters())

    def reconstruct(self, belief_params: TensorDict) -> TensorDict:
        return self.net(belief_params)


def set_self_condition(params: TensorDict, rec: TensorDict) -> None:
    params["X_sc"] = rec["X"].detach().clone()
    params["E_sc"] = rec["E"].detach().clone()


def _simplex(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp_min(0)
    return x / x.sum(-1, keepdim=True).clamp_min(1e-12)


def _extrapolate(hist: list[tuple[float, dict[str, torch.Tensor]]], key: str, offset: float) -> torch.Tensor:
    """Newton forward extrapolation of the reconstruction to t_last + offset.

    Two history entries give the linear (Adams-Bashforth-2-like) extrapolation,
    three give the quadratic one. Both reuse reconstructions already computed.
    """
    ts = [t for t, _ in hist]
    xs = [d[key] for _, d in hist]
    t_eval = ts[-1] + offset
    # Newton's divided differences, built in place.
    coeffs = list(xs)
    for level in range(1, len(xs)):
        for i in range(len(xs) - 1, level - 1, -1):
            coeffs[i] = (coeffs[i] - coeffs[i - 1]) / (ts[i] - ts[i - level])
    out = coeffs[0]
    basis = 1.0
    for i in range(1, len(xs)):
        basis = basis * (t_eval - ts[i - 1])
        out = out + coeffs[i] * basis
    return out


def _temper(probs: torch.Tensor, tau: float) -> torch.Tensor:
    if tau == 1.0:
        return probs
    logp = torch.log(probs.clamp_min(1e-30)) / tau
    out = torch.softmax(logp, dim=-1)
    return torch.where(probs > 0, out, torch.zeros_like(out))


def timesteps(cfg: SamplerConfig) -> torch.Tensor:
    # identical to Sampler.get_timesteps with (nfe - 1) grid points
    n = cfg.nfe - 1
    t = torch.linspace(0, 1, n)
    return (t * cfg.t_max ** (1 / cfg.rho)) ** cfg.rho


@torch.no_grad()
def sample_batch(model: Model, B: int, cfg: SamplerConfig) -> TensorDict:
    belief = model.belief
    dev = model.device
    params = belief.sample_initial_belief(B)
    params["t"] = torch.zeros(B, device=dev)
    cls = model.class_conditioner.sample_classifications(B, device=dev)
    model.class_conditioner.write_sample_classifications_to_belief(cls, params)
    prior = params.clone()
    ts = timesteps(cfg)
    hist: list[tuple[float, dict[str, torch.Tensor]]] = []
    for t1, t2 in pairwise(ts):
        t1f = float(t1)
        params["t"] = torch.full((B,), t1f, device=dev)
        rec = model.reconstruct(params)
        if model.selfcond and cfg.selfcond:
            set_self_condition(params, rec)
        if t1f >= cfg.tau_from * cfg.t_max:
            rec["X"] = _temper(rec["X"], cfg.tau_x)
            rec["E"] = _temper(rec["E"], cfg.tau_e)
        h = float(t2 - t1)
        if cfg.bg != 0.0 and cfg.bg_lo <= t1f / cfg.t_max <= cfg.bg_hi:
            for k in ("X", "E"):
                rec[k] = _simplex(rec[k] + cfg.bg * (rec[k] - torch.softmax(params[k], -1)))
        if cfg.ms != 0.0:
            hist.append((t1f, {k: rec[k].clone() for k in ("X", "E")}))
            hist[:] = hist[-cfg.ms_order :]
            if len(hist) == cfg.ms_order and cfg.ms_lo <= t1f / cfg.t_max <= cfg.ms_hi:
                for k in ("X", "E"):
                    w = cfg.ms * (cfg.ms_x if k == "X" else cfg.ms_e)
                    x = _extrapolate(hist, k, w * h)
                    if cfg.ms_clip:
                        x = _simplex(x)
                    rec[k] = x
        gamma, method = cfg.gamma, cfg.method
        if cfg.method2 is not None and t1f / cfg.t_max >= cfg.switch_at:
            method = cfg.method2
            if cfg.gamma2 is not None:
                gamma = cfg.gamma2
        if cfg.gamma_end is not None:
            gamma = gamma * (cfg.gamma_end / cfg.gamma) ** (t1f / cfg.t_max)
        belief.discretize_belief_update_inplace(
            params, rec, prior, (t2 - t1).to(dev), gamma, False, method=method
        )
    t_final = float(ts[-1]) if cfg.final_t == "end" else float(ts[-2])
    params["t"] = torch.full((B,), t_final, device=dev)
    final = model.reconstruct(params)
    model.class_conditioner.write_sample_classifications_to_belief(cls, final)
    if cfg.final_decode == "sample":
        for k in ("X", "E"):
            p = final[k]
            idx = torch.distributions.Categorical(probs=p.clamp_min(0) + 1e-12).sample()
            final[k] = torch.nn.functional.one_hot(idx, p.shape[-1]).float()
        # keep E symmetric
        E = final["E"].argmax(-1)
        E = torch.triu(E, 1)
        E = E + E.transpose(1, 2)
        final["E"] = torch.nn.functional.one_hot(E, final["E"].shape[-1]).float()
    return final


def to_smiles(model: Model, batch: TensorDict) -> list[Optional[str]]:
    from rdkit import Chem

    out: list[Optional[str]] = []
    batch = batch.to("cpu")
    for mol in batch:
        m = model.datamodule.molecular_sample_to_molecule(mol)
        out.append(Chem.MolToSmiles(m) if m is not None else None)
    return out


def sample_smiles(model: Model, n: int, cfg: SamplerConfig) -> list[Optional[str]]:
    """Sample n SMILES, halving the batch size when the GPU runs out of memory.

    GuacaMol graphs are large enough that the batch that fits on an 80 GB card
    does not fit on a 40 GB one, and losing hours of sampling to that is not
    worth the saved memory.
    """
    torch.manual_seed(cfg.seed)
    smiles: list[Optional[str]] = []
    batch = cfg.batch_size
    while len(smiles) < n:
        b = min(batch, n - len(smiles))
        try:
            smiles += to_smiles(model, sample_batch(model, b, cfg))
        except torch.OutOfMemoryError:
            if b <= 16:
                raise
            batch = b // 2
            torch.cuda.empty_cache()
            print(f"out of memory at batch {b}, retrying with {batch}", flush=True)
    return smiles


# ----------------------------------------------------------------------------- scoring

BENCHMARKER = {
    "moses": "data/molecules/moses/benchmarker_0213048b3ced5eb7cc0c4702b834f907_ed7eb093bf74ff81bcff185a1900b2fd_25000.pkl",
    "guacamol": "data/molecules/guacamol/benchmarker_f36442b7434eb4f8d568862de1b7aad8_bad6af49aae36caa0b0f93bb01f18f06_15000.pkl",
}


def load_benchmarker(dataset: str):
    with open(REPO / BENCHMARKER[dataset], "rb") as f:
        return pickle.load(f)


def headline(dataset: str, flat: dict) -> dict:
    """Map benchmark output to the Table 1 columns (same mapping as scripts/summarize.py)."""
    v = lambda k: flat.get(k)  # noqa: E731
    if dataset == "moses":
        return {
            "validity": 100 * v("validity/valid_fraction"),
            "unique": 100 * v("validity/unique_fraction_at_1000"),
            "novel": 100 * v("validity/unique_and_novel_fraction"),
            "filters": 100 * v("moses/fraction_passing_moses_filters"),
            "fcd": v("fcd/fcd_valid"),
            "snn": v("moses/snn_score"),
            "scaffold": 100 * v("moses/scaffolds_similarity"),
        }
    return {
        "validity": 100 * v("validity/valid_fraction"),
        "valid_unique": 100 * v("validity/valid_and_unique_fraction"),
        "valid_unique_novel": 100 * v("validity/valid_and_unique_and_novel_fraction"),
        "kl": 100 * v("kl_score"),
        "fcd": 100 * v("fcd/fcd_valid_normalized"),
    }


def score_smiles(dataset: str, smiles: list[Optional[str]], device: str = "cpu", bm=None) -> dict:
    from project.plots.molecule import flatten_dict

    bm = bm or load_benchmarker(dataset)
    bm.num_samples_to_generate = len(smiles)
    bm.device = device
    flat = flatten_dict(bm.benchmark(smiles))
    return {"headline": headline(dataset, flat), "all": flat}


def save(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    tmp.replace(path)


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def fmt(h: dict) -> str:
    return " ".join(f"{k}={val:.3f}" if isinstance(val, float) and not math.isnan(val) else f"{k}={val}" for k, val in h.items())
