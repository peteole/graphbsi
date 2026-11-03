"""Score a run directory written by sweep.py: <run>/smiles.json -> <run>/metrics.json."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import fmt, save, score_smiles  # noqa: E402

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    for run in map(Path, a.runs):
        meta = json.loads((run / "config.json").read_text())
        smiles = json.loads((run / "smiles.json").read_text())
        res = score_smiles(meta["dataset"], smiles, device=a.device)
        res["config"] = meta
        save(run / "metrics.json", res)
        print(run.name, fmt(res["headline"]), flush=True)
