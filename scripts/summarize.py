#!/usr/bin/env python
"""Turn the per-row JSON files in ``results/`` into paper-comparison tables."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from paper_values import TABLE1, TABLE2  # noqa: E402

# How far a reproduced value may deviate from the published one before it is
# flagged.  Molecule metrics are averaged over 15k-25k samples and are very
# stable; the synthetic-graph numbers come from only 5 batches of 40 graphs, so
# their tolerance is two standard errors of the difference of two such means.
# With the paper run's seed, environment and GPU the values are not merely
# close but identical; such cells are marked as exact.
MOLECULE_TOLERANCE = {
    "validity": 0.5,
    "valid_unique": 0.5,
    "valid_unique_novel": 0.5,
    "unique": 0.5,
    "novel": 1.0,
    "filters": 0.5,
    "kl": 1.0,
    "fcd": 1.5,  # GuacaMol: normalized FCD in %
    "snn": 0.02,
    "scaffold": 3.0,
}
MOSES_FCD_TOLERANCE = 0.10  # raw FCD
GRAPH_BATCHES = 5

# metric key -> (suffix of the captured metric name, scale)
MOLECULE_METRICS = {
    "validity": ("validity/valid_fraction", 100.0),
    "valid_unique": ("validity/valid_and_unique_fraction", 100.0),
    "valid_unique_novel": ("validity/valid_and_unique_and_novel_fraction", 100.0),
    "unique": ("validity/unique_fraction_at_1000", 100.0),
    "novel": ("validity/unique_and_novel_fraction", 100.0),  # Tab. 1 "Nov."
    "filters": ("moses/fraction_passing_moses_filters", 100.0),
    "snn": ("moses/snn_score", 1.0),
    "scaffold": ("moses/scaffolds_similarity", 100.0),
}
MOLECULE_HEADERS = {
    "guacamol": [
        ("validity", "Val. %", 1),
        ("valid_unique", "V.U. %", 1),
        ("valid_unique_novel", "V.U.N. %", 1),
        ("kl", "KL %", 1),
        ("fcd", "FCD %", 1),
    ],
    "moses": [
        ("validity", "Val. %", 1),
        ("unique", "Uniq. %", 1),
        ("novel", "Nov. %", 1),
        ("filters", "Filters %", 1),
        ("fcd", "FCD", 2),
        ("snn", "SNN", 2),
        ("scaffold", "Scaf. %", 1),
    ],
}
ROW_TITLES = {
    "em_50": "GraphBSI (EM), 50 steps",
    "ou_50": "GraphBSI (OU), 50 steps",
    "em_500": "GraphBSI (EM), 500 steps",
    "ou_500": "GraphBSI (OU), 500 steps",
    "em_1000": "GraphBSI (EM), 1000 steps",
    "ou_1000": "GraphBSI (OU), 1000 steps",
}


def _lookup(metrics: dict[str, float], suffix: str, *, std: bool = False) -> float | None:
    for key, value in metrics.items():
        if not key.endswith("/" + suffix):
            continue
        if ("_std/" in key) != std:
            continue
        return value
    return None


def molecule_values(dataset: str, metrics: dict[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for name, (suffix, scale) in MOLECULE_METRICS.items():
        value = _lookup(metrics, suffix)
        if value is not None:
            out[name] = value * scale
    kl = _lookup(metrics, "kl_score")
    if kl is not None:
        out["kl"] = kl * 100.0
    if dataset == "guacamol":
        fcd = _lookup(metrics, "fcd/fcd_valid_normalized")
        if fcd is not None:
            out["fcd"] = fcd * 100.0
    else:
        fcd = _lookup(metrics, "fcd/fcd_valid")
        if fcd is not None:
            out["fcd"] = fcd
    return out


def graph_values(dataset: str, metrics: dict[str, float]) -> dict[str, tuple[float, float]]:
    out: dict[str, tuple[float, float]] = {}
    # The paper's V.U. column is the benchmark's validity accuracy
    # (`<dataset>_acc`), not `sampling/frac_unic_non_iso_valid`; the two
    # coincide for planar and tree but not for SBM.
    vu = _lookup(metrics, f"{dataset}_acc")
    vu_std = _lookup(metrics, f"{dataset}_acc", std=True)
    if vu is not None:
        out["vu"] = (vu * 100.0, (vu_std or 0.0) * 100.0)
    ratio = _lookup(metrics, "average_ratio")
    ratio_std = _lookup(metrics, "average_ratio", std=True)
    if ratio is not None:
        out["ratio"] = (ratio, ratio_std or 0.0)
    return out


def load_results(out_dir: Path) -> dict[tuple[str, str], dict[str, Any]]:
    results: dict[tuple[str, str], dict[str, Any]] = {}
    for path in sorted(out_dir.glob("*.json")):
        if path.name in {"all_results.json", "comparison.json"}:
            continue
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict) or "dataset" not in data:
            continue
        results[(data["dataset"], data.get("label", ""))] = data
    return results


def _fmt(value: float | None, digits: int) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _exact(value: float | None, paper: float, digits: int) -> bool:
    """Whether `value` prints as the published number."""
    return value is not None and f"{value:.{digits}f}" == f"{paper:.{digits}f}"


def _mark(ok: bool, exact: bool) -> str:
    return " ✓" if exact else ("" if ok else " ⚠")


def _table1_section(results, checks) -> list[str]:
    lines = ["## Table 1 - molecule generation", ""]
    for dataset in ("guacamol", "moses"):
        headers = MOLECULE_HEADERS[dataset]
        lines.append(f"### {dataset}")
        lines.append("")
        lines.append(
            "| Configuration | Source | " + " | ".join(h[1] for h in headers) + " |"
        )
        lines.append("| --- | --- | " + " | ".join("---" for _ in headers) + " |")
        for label in ("em_50", "ou_50", "em_500", "ou_500"):
            paper = TABLE1.get((dataset, label))
            if paper is None:
                continue
            data = results.get((dataset, label))
            title = ROW_TITLES.get(label, label)
            lines.append(
                f"| {title} | paper | "
                + " | ".join(_fmt(paper.get(k), d) for k, _, d in headers)
                + " |"
            )
            if data is None:
                for key, _, _digits in headers:
                    checks.append(
                        dict(
                            table="Table 1",
                            row=f"{dataset}:{label}",
                            metric=key,
                            paper=paper[key],
                            reproduced=None,
                            tolerance=None,
                            ok=False,
                        )
                    )
                lines.append(
                    f"| {title} | reproduced | " + " | ".join("-" for _ in headers) + " |"
                )
                continue
            got = molecule_values(dataset, data["metrics"])
            cells = []
            for key, _, digits in headers:
                value = got.get(key)
                tol = MOSES_FCD_TOLERANCE if (dataset == "moses" and key == "fcd") else MOLECULE_TOLERANCE[key]
                ok = value is not None and abs(value - paper[key]) <= tol
                exact = _exact(value, paper[key], digits)
                checks.append(
                    dict(
                        table="Table 1",
                        row=f"{dataset}:{label}",
                        metric=key,
                        paper=paper[key],
                        reproduced=value,
                        tolerance=tol,
                        ok=bool(ok),
                        exact=exact,
                    )
                )
                cells.append(_fmt(value, digits) + _mark(ok, exact))
            lines.append(f"| {title} | reproduced | " + " | ".join(cells) + " |")
        lines.append("")
    return lines


def _table2_section(results, checks) -> list[str]:
    lines = ["## Table 2 - synthetic graph generation", ""]
    datasets = ("planar", "tree", "sbm")
    lines.append(
        "| Configuration | Source | "
        + " | ".join(f"{d} V.U. % | {d} ratio" for d in datasets)
        + " |"
    )
    lines.append("| --- | --- | " + " | ".join("---" for _ in datasets for _ in (0, 1)) + " |")
    for label in ("em_50", "ou_50", "em_1000", "ou_1000"):
        title = ROW_TITLES.get(label, label)
        paper_cells, repro_cells = [], []
        for dataset in datasets:
            paper = TABLE2.get((dataset, label))
            data = results.get((dataset, label))
            if paper is None:
                paper_cells += ["-", "-"]
            else:
                paper_cells += [
                    f"{paper['vu'][0]:.1f} ± {paper['vu'][1]:.1f}",
                    f"{paper['ratio'][0]:.1f} ± {paper['ratio'][1]:.1f}",
                ]
            if data is None or paper is None:
                if paper is not None:
                    for key in ("vu", "ratio"):
                        checks.append(
                            dict(
                                table="Table 2",
                                row=f"{dataset}:{label}",
                                metric=key,
                                paper=paper[key][0],
                                reproduced=None,
                                tolerance=None,
                                ok=False,
                            )
                        )
                repro_cells += ["-", "-"]
                continue
            got = graph_values(dataset, data["metrics"])
            for key, digits in (("vu", 1), ("ratio", 2)):
                value = got.get(key)
                # Two standard errors of the difference of two means over
                # GRAPH_BATCHES batches each; the reported +- is the spread of
                # the per-batch values.
                spread = (paper[key][1] ** 2 + (value[1] if value else 0.0) ** 2) ** 0.5
                tol = max(2 * spread / GRAPH_BATCHES**0.5, 0.5 if key == "vu" else 0.1)
                ok = value is not None and abs(value[0] - paper[key][0]) <= tol
                exact = value is not None and _exact(value[0], paper[key][0], 1)
                checks.append(
                    dict(
                        table="Table 2",
                        row=f"{dataset}:{label}",
                        metric=key,
                        paper=paper[key][0],
                        reproduced=None if value is None else value[0],
                        tolerance=tol,
                        ok=bool(ok),
                        exact=exact,
                    )
                )
                if value is None:
                    repro_cells.append("-")
                else:
                    repro_cells.append(
                        f"{value[0]:.{digits}f} ± {value[1]:.{digits}f}" + _mark(ok, exact)
                    )
        lines.append(f"| {title} | paper | " + " | ".join(paper_cells) + " |")
        lines.append(f"| {title} | reproduced | " + " | ".join(repro_cells) + " |")
    lines.append("")
    return lines


def _other_section(results) -> list[str]:
    others = {
        key: value
        for key, value in results.items()
        if value.get("paper_table") not in ("Table 1", "Table 2")
    }
    if not others:
        return []
    lines = ["## Other tables", ""]
    for (dataset, label), data in sorted(others.items()):
        lines.append(f"### {dataset} / {label} ({data.get('paper_table')})")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("| --- | --- |")
        for key in sorted(data["metrics"]):
            lines.append(f"| `{key}` | {data['metrics'][key]:.6g} |")
        lines.append("")
    return lines


def write_reports(out_dir: Path) -> dict[str, Any]:
    results = load_results(out_dir)
    checks: list[dict[str, Any]] = []
    lines = ["# GraphBSI reproduction results", ""]
    quick = any(r.get("quick") for r in results.values())
    if quick:
        lines += ["> ⚠ Some rows were produced in `--quick` mode and are **not**", "> comparable to the paper.", ""]
    lines += _table1_section(results, checks)
    lines += _table2_section(results, checks)
    lines += _other_section(results)

    done = [c for c in checks if c["reproduced"] is not None]
    failed = [c for c in done if not c["ok"]]
    summary = {
        "checked": len(done),
        "exact": sum(1 for c in done if c.get("exact")),
        "matched": len(done) - len(failed),
        "missing": len(checks) - len(done),
        "failed": failed,
        "quick": quick,
    }
    lines += [
        "## Summary",
        "",
        f"- Metrics compared to the paper: **{summary['checked']}**",
        f"- Identical to the paper at its printed precision (✓): **{summary['exact']}**",
        f"- Within tolerance: **{summary['matched']}**",
        f"- Outside tolerance: **{len(failed)}**",
        f"- Not run: **{summary['missing']}**",
        "",
    ]
    if failed:
        lines += ["| Row | Metric | Paper | Reproduced | Tolerance |", "| --- | --- | --- | --- | --- |"]
        for c in failed:
            lines.append(
                f"| {c['row']} | {c['metric']} | {c['paper']:.3g} | {c['reproduced']:.3g} | ±{c['tolerance']:.3g} |"
            )
        lines.append("")

    (out_dir / "metrics.md").write_text("\n".join(lines))
    (out_dir / "comparison.json").write_text(
        json.dumps({"summary": summary, "checks": checks}, indent=2)
    )
    (out_dir / "all_results.json").write_text(
        json.dumps(sorted(results.values(), key=lambda d: (d["dataset"], d.get("label", ""))), indent=2)
    )
    print("\n" + "\n".join(lines[-12:]))
    print(f"\nSummary: {out_dir / 'metrics.md'}")
    return summary


def main() -> None:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO_ROOT / "results"
    write_reports(out_dir)


if __name__ == "__main__":
    main()
