"""Format scored runs as a markdown table next to the paper and the best
published numbers.

usage: python research/report.py RUN_DIR [RUN_DIR ...] > research/results.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.paper_values import TABLE1  # noqa: E402

# Best published value per metric and who holds it (see research/sota_metrics.md).
# "graph" = best atom-level one-shot graph model, "any" = best of any model type.
SOTA = {
    "moses": {
        "fcd": (0.72, "GraphBSI", 0.52, "CharRNN (SMILES)"),
        "scaffold": (23.5, "DeFoG@50", 23.5, "DeFoG@50"),
        "novel": (100.0, "DMol", 100.0, "DMol"),
        "validity": (99.9, "GraphBSI", 100.0, "JT-VAE"),
        "filters": (99.2, "GraphBSI", 99.7, "VAE"),
        "snn": (0.59, "GraphBFN", 0.59, "GraphBFN"),
        "unique": (100.0, "several", 100.0, "several"),
    },
    "guacamol": {
        "fcd": (82.6, "GraphBSI", 93.4, "G2PT-base (85M)"),
        "kl": (98.4, "GraphBSI", 99.6, "GVT"),
        "validity": (99.7, "FragFM", 100.0, "Graph MCTS"),
        "valid_unique": (99.6, "GraphBSI", 100.0, "G2PT"),
        "valid_unique_novel": (98.7, "GraphBSI", 98.7, "GraphBSI"),
    },
}

COLUMNS = {
    "moses": [("validity", "Val.", 1), ("unique", "Uniq.", 1), ("novel", "Nov.", 1),
              ("filters", "Filters", 1), ("fcd", "FCD", 3), ("snn", "SNN", 3),
              ("scaffold", "Scaf.", 1)],
    "guacamol": [("validity", "Val.", 1), ("valid_unique", "V.U.", 1),
                 ("valid_unique_novel", "V.U.N.", 1), ("kl", "KL", 1), ("fcd", "FCD", 1)],
}
LOWER_BETTER = {("moses", "fcd")}


def load(dirs: list[Path]) -> list[dict]:
    runs = []
    for root in dirs:
        for m in sorted(root.rglob("metrics.json")):
            r = json.loads(m.read_text())
            cfg = r.get("config") or json.loads((m.parent / "config.json").read_text())
            runs.append({"name": m.parent.name, "dir": str(m.parent), "dataset": cfg["dataset"],
                         "n": cfg["n"], "sampler": cfg["sampler"], "headline": r["headline"]})
    return runs


def fmt(v: float | None, digits: int) -> str:
    return "-" if v is None else f"{v:.{digits}f}"


def main(dirs: list[Path]) -> None:
    runs = load(dirs)
    for dataset in ("moses", "guacamol"):
        rows = [r for r in runs if r["dataset"] == dataset]
        if not rows:
            continue
        cols = COLUMNS[dataset]
        print(f"\n## {dataset}\n")
        head = ["Sampler", "NFE", "n"] + [c[1] for c in cols]
        print("| " + " | ".join(head) + " |")
        print("|" + "---|" * len(head))
        for r in sorted(rows, key=lambda r: (r["sampler"]["nfe"], r["name"])):
            s = r["sampler"]
            cells = [fmt(r["headline"].get(k), d) for k, _, d in cols]
            print(f"| {r['name']} | {s['nfe']} | {r['n']} | " + " | ".join(cells) + " |")
        # paper rows for reference
        for (ds, label), vals in TABLE1.items():
            if ds != dataset:
                continue
            cells = [fmt(vals.get(k), d) for k, _, d in cols]
            n_paper = 25000 if dataset == "moses" else 15000
            print(f"| *paper {label}* | {label.split('_')[-1]} | {n_paper} | " + " | ".join(cells) + " |")
        print("\nBest published, for reference:\n")
        print("| Metric | Best atom-level graph model | Best any model |")
        print("|---|---|---|")
        for k, label, d in cols:
            if k not in SOTA[dataset]:
                continue
            g, gw, a, aw = SOTA[dataset][k]
            arrow = "↓" if (dataset, k) in LOWER_BETTER else "↑"
            print(f"| {label} {arrow} | {fmt(g, d)} ({gw}) | {fmt(a, d)} ({aw}) |")


if __name__ == "__main__":
    main([Path(p) for p in sys.argv[1:]])
