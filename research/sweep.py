"""Sample a list of sampler configs with one model load; score each in a
background subprocess while the GPU moves on to the next config.

usage: python research/sweep.py --dataset moses --n 10000 --out RUNS/<sweep> \
          --grid '{"nfe":[50],"method":["euler-maruyama"],"gamma":[5,10,20]}'
"""

import argparse
import itertools
import json
import subprocess
import sys
import time
from dataclasses import asdict, fields
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import Model, SamplerConfig, now, sample_smiles, save  # noqa: E402

HERE = Path(__file__).resolve().parent


def expand(grid: dict) -> list[dict]:
    keys = list(grid)
    vals = [v if isinstance(v, list) else [v] for v in grid.values()]
    return [dict(zip(keys, combo)) for combo in itertools.product(*vals)]


def run_name(d: dict) -> str:
    short = {"euler-maruyama": "em", "ornstein-uhlenbeck": "ou"}
    parts = []
    for k, v in d.items():
        if k == "extra":
            continue
        v = short.get(v, v)
        parts.append(f"{k}={v}")
    return ",".join(parts) or "default"


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--n", type=int, default=10000)
    p.add_argument("--out", required=True)
    p.add_argument("--grid", action="append", default=[], help="JSON dict of lists; may repeat")
    p.add_argument("--configs", default=None, help="JSON file with a list of config dicts")
    p.add_argument("--max-scorers", type=int, default=2)
    p.add_argument("--score-device", default="cpu")
    a = p.parse_args()

    configs: list[dict] = []
    for g in a.grid:
        configs += expand(json.loads(g))
    if a.configs:
        configs += json.loads(Path(a.configs).read_text())
    valid = {f.name for f in fields(SamplerConfig)}
    for c in configs:
        bad = set(c) - valid
        assert not bad, f"unknown keys {bad}"

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model = Model(a.dataset, a.ckpt)
    print(f"[{now()}] loaded {model.ckpt}, params={model.num_parameters()}", flush=True)
    procs: list[subprocess.Popen] = []
    for c in configs:
        cfg = SamplerConfig(**c)
        run = out / run_name(c)
        if (run / "metrics.json").exists() or (run / "smiles.json").exists():
            print(f"skip {run.name}", flush=True)
            continue
        t0 = time.time()
        smiles = sample_smiles(model, a.n, cfg)
        meta = {"dataset": a.dataset, "ckpt": model.ckpt, "n": a.n, "sampler": asdict(cfg),
                "sample_seconds": time.time() - t0, "overrides": c}
        save(run / "config.json", meta)
        save(run / "smiles.json", smiles)
        print(f"[{now()}] sampled {run.name} in {time.time() - t0:.0f}s", flush=True)
        if a.max_scorers == 0:
            continue
        while len([q for q in procs if q.poll() is None]) >= a.max_scorers:
            time.sleep(5)
        log = open(run / "score.log", "w")
        procs.append(subprocess.Popen(
            [sys.executable, str(HERE / "score.py"), str(run), "--device", a.score_device],
            stdout=log, stderr=subprocess.STDOUT))
    for q in procs:
        q.wait()
    for c in configs:
        m = out / run_name(c) / "metrics.json"
        if m.exists():
            h = json.loads(m.read_text())["headline"]
            print(run_name(c), " ".join(f"{k}={v:.3f}" for k, v in h.items()), flush=True)
