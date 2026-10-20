#!/usr/bin/env python
"""Reproduce GraphBSI paper metrics from the released checkpoints.

Every *row* of Tabs. 1 and 2 in the paper corresponds to one entry in ``ROWS``
below: a dataset, a discretization scheme (Euler-Maruyama / Ornstein-Uhlenbeck),
a number of function evaluations and the noise level ``gamma`` that was tuned
for that configuration.  For each row we run ``train.py`` in
validate-from-checkpoint mode, capture the scalar metrics that the plot
callbacks normally send to wandb, and write JSON + markdown summaries under
``results/``.

Each row also carries the seed of the wandb run that produced the published
number.  Sampling is deterministic given the seed, the sampling batch size, the
environment and the GPU architecture, so with all four matching the paper
numbers come out exactly (see ``REPRODUCTION.md``).

Examples:
  python scripts/reproduce.py --tables 1 2          # everything in Tabs. 1 + 2
  python scripts/reproduce.py --datasets planar     # one dataset, all rows
  python scripts/reproduce.py --rows moses:ou_500
  python scripts/reproduce.py --all --quick         # smoke test
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from summarize import write_reports  # noqa: E402

TOY_ALPHABETS = [5, 10, 20, 40, 60, 80, 100, 120, 140, 160]

EM = "euler-maruyama"
OU = "ornstein-uhlenbeck"

#: The two locked environments. The paper's runs used the uv one (torch 2.7,
#: Python 3.12) except for the SBM ablations, which ran in the pixi one (torch
#: 2.8, Python 3.13) because the SBM metric needs graph-tool from conda-forge.
ENV_PYTHON = {
    "uv": REPO_ROOT / ".venv" / "bin" / "python",
    "pixi": REPO_ROOT / ".pixi" / "envs" / "default" / "bin" / "python",
}

GTX1080TI = "GTX 1080 Ti"
RTX2080TI = "RTX 2080 Ti"
A100 = "A100"


@dataclass(frozen=True)
class Row:
    """One row of a paper table."""

    dataset: str
    label: str
    method: str
    gamma: float
    nfe: int
    table: str
    experiment: str
    checkpoint: str | None
    num_samples: int | None = None
    extra: tuple[str, ...] = ()
    quick: tuple[str, ...] = ()
    batch_size_key: str | None = None
    default_batch_size: int | None = None
    #: seed of the wandb run behind the published number (None: not replayable)
    seed: str | None = None
    #: that wandb run, as project/run_id
    source: str | None = None
    env: str = "uv"
    #: GPU architecture of the paper run; sampling is only bit-exact on it
    gpu: str | None = None

    @property
    def key(self) -> str:
        return f"{self.dataset}:{self.label}"


#: label -> (method, gamma, nfe, seed, source run, GPU of the source run)
Spec = tuple[str, float, int, str | None, str, str]


def _molecule_rows(
    dataset: str,
    num_samples: int,
    specs: dict[str, Spec],
    checkpoints: dict[str, str] | None = None,
    extra: dict[str, tuple[str, ...]] | None = None,
) -> list[Row]:
    rows = []
    for label, (method, gamma, nfe, seed, source, gpu) in specs.items():
        rows.append(
            Row(
                dataset=dataset,
                label=label,
                method=method,
                gamma=gamma,
                nfe=nfe,
                seed=seed,
                source=source,
                gpu=gpu,
                table="Table 1",
                experiment=f"{dataset}/reproduce",
                checkpoint=(checkpoints or {}).get(
                    label, f"final_checkpoints/{dataset}/last.ckpt"
                ),
                num_samples=num_samples,
                extra=(extra or {}).get(label, ()),
                batch_size_key="plots.0.batch_size",
                quick=(
                    "plots.0.num_samples=256",
                    "plots.0.batch_size=64",
                    "task.sampler.variable_noise.max_function_evals=20",
                ),
            )
        )
    return rows


def _graph_rows(dataset: str, specs: dict[str, Spec], env: str = "uv") -> list[Row]:
    rows = []
    for label, (method, gamma, nfe, seed, source, gpu) in specs.items():
        rows.append(
            Row(
                dataset=dataset,
                label=label,
                method=method,
                gamma=gamma,
                nfe=nfe,
                seed=seed,
                source=source,
                gpu=gpu,
                env=env,
                table="Table 2",
                experiment=f"{dataset}/reproduce",
                checkpoint=f"final_checkpoints/{dataset}/last.ckpt",
                # SBM graphs have up to 187 nodes and the (bs, n, n, heads, df)
                # attention tensor is the memory bottleneck; halved on OOM.
                batch_size_key="plots.2.batch_size",
                default_batch_size=16,
                quick=(
                    "plots.2.num_samples=40",
                    "plots.2.metric_batch_size=40",
                    "task.sampler.variable_noise.max_function_evals=50",
                ),
            )
        )
    return rows


# Noise levels, seeds and GPUs are those of the wandb runs behind the published
# numbers (entity peteole-technical-university-of-munich).
MOSES_RUNS = "molecules_ablation_12_layers_moses"
GUACAMOL_RUNS = "molecules_ablation_12_layers_guacamol"
ROWS: list[Row] = [
    *_molecule_rows(
        "moses",
        25_000,
        {
            "em_50": (EM, 10.0, 50, "214368324826129980872585727243579172111", f"{MOSES_RUNS}/7zrha335", GTX1080TI),
            "ou_50": (OU, 20.0, 50, "259448201183481664105574468257011701260", f"{MOSES_RUNS}/6mov362j", GTX1080TI),
            "em_500": (EM, 100.0, 500, "249393440397453012479444854866603313689", f"{MOSES_RUNS}/7q53x74d", GTX1080TI),
            "ou_500": (OU, 90.0, 500, "229378774089056598278205302455704210200", f"{MOSES_RUNS}/mlsggyxe", GTX1080TI),
        },
        # The 500-step paper runs built the benchmark reference after seeding,
        # which draws random numbers, so these rows build it as well.
        extra={
            "em_500": ("+plots.0.rebuild_benchmarker=true",),
            "ou_500": ("+plots.0.rebuild_benchmarker=true",),
        },
    ),
    *_molecule_rows(
        "guacamol",
        15_000,
        {
            "em_50": (EM, 10.0, 50, "297496729413224878188687548319076444123", f"{GUACAMOL_RUNS}/o9pvyeps", A100),
            "ou_50": (OU, 20.0, 50, "78549004695834586247171698364956321206", f"{GUACAMOL_RUNS}/sp33mijc", A100),
            "em_500": (EM, 200.0, 500, "75701702256067293249906264555029935762", f"{GUACAMOL_RUNS}/y264r6rd", A100),
            "ou_500": (OU, 250.0, 500, "59015920368729694425502593363210102833", f"{GUACAMOL_RUNS}/ejp3imis", A100),
        },
        # GuacaMol training was continued across several runs. The 50-step Euler
        # row was evaluated while run 17oqxx3h was still training and loaded its
        # epoch-59 checkpoint; the 500-step rows used the final epoch-89 one
        # (`last.ckpt`). The 50-step OU row loaded an epoch-69 checkpoint of run
        # 635bennr that training later overwrote, so it runs on `last.ckpt`.
        checkpoints={"em_50": "final_checkpoints/guacamol/epoch_59.ckpt"},
    ),
    *_graph_rows(
        "planar",
        {
            "em_50": (EM, 20.0, 50, "126175628054956385461699258598608172808", "ablations_planar/8q2dpcx6", GTX1080TI),
            "ou_50": (OU, 40.0, 50, "205142485125591307548003176387054845796", "ablations_planar/bti8kpmr", GTX1080TI),
            "em_1000": (EM, 200.0, 1000, "228815049410501089291504653704180089869", "ablations_planar/enf9hwf3", GTX1080TI),
            # Logged by the validation loop of the training run itself; its RNG
            # state at that point cannot be replayed from a checkpoint.
            "ou_1000": (OU, 200.0, 1000, None, "train_planar/agag753c", A100),
        },
    ),
    *_graph_rows(
        "tree",
        {
            "em_50": (EM, 30.0, 50, "195051016455221298984181017717337153691", "ablations_tree/lvjkewm6", GTX1080TI),
            "ou_50": (OU, 30.0, 50, "127030285702017813380880924221858542182", "ablations_tree/q9wb7nvs", GTX1080TI),
            "em_1000": (EM, 175.0, 1000, "119407846801454450466620158060816151854", "ablations_tree/w35zc7cq", A100),
            "ou_1000": (OU, 500.0, 1000, "154521081622716558052391601363952025158", "ablations_tree/ut4wllb4", A100),
        },
    ),
    *_graph_rows(
        "sbm",
        {
            "em_50": (EM, 15.0, 50, "178893730595251566962404736909858043381", "ablations_sbm/qdmayuv3", RTX2080TI),
            "ou_50": (OU, 15.0, 50, "70099103804785947213864327735616268777", "ablations_sbm/hij7alnp", RTX2080TI),
            "em_1000": (EM, 200.0, 1000, "123248381148823760130313707874366896163", "ablations_sbm/mmy3zxio", RTX2080TI),
            # Logged by the validation loop of the training run itself (step
            # 28000, the released checkpoint); not replayable, see above.
            "ou_1000": (OU, 200.0, 1000, None, "train_sbm/aog3lu3i", A100),
        },
        env="pixi",
    ),
    # Appendix / other tables. Not part of --tables 1 2.
    Row(
        dataset="qm9",
        label="ou_500",
        method=OU,
        gamma=90.0,
        nfe=500,
        table="Table 6",
        experiment="qm9/reproduce",
        checkpoint="final_checkpoints/qm9/last.ckpt",
        quick=("plots.0.num_samples=256", "task.sampler.variable_noise.max_function_evals=20"),
    ),
    Row(
        dataset="qm9_with_h",
        label="ou_500",
        method=OU,
        gamma=90.0,
        nfe=500,
        table="Table 6",
        experiment="qm9_with_h/reproduce",
        checkpoint="final_checkpoints/qm9_with_h/last.ckpt",
        quick=("plots.0.num_samples=256", "task.sampler.variable_noise.max_function_evals=20"),
    ),
    Row(
        dataset="flybrain",
        label="ou_100",
        method=OU,
        gamma=5.0,
        nfe=100,
        table="Table 3",
        experiment="flybrain/reproduce",
        checkpoint="final_checkpoints/flybrain/last.ckpt",
        quick=("plots.0.num_samples=256", "task.sampler.variable_noise.max_function_evals=20"),
    ),
    Row(
        dataset="mel2",
        label="ou_100",
        method=OU,
        gamma=10.0,
        nfe=100,
        table="Table 3",
        experiment="mel2/reproduce",
        checkpoint="final_checkpoints/mel2/last.ckpt",
        quick=("plots.0.num_samples=256", "task.sampler.variable_noise.max_function_evals=20"),
    ),
    *[
        Row(
            dataset="toy_sequence",
            label=f"alphabet_{a}",
            method=OU,
            gamma=5.0,
            nfe=100,
            table="Figure 9",
            experiment="toy_sequence/reproduce",
            checkpoint=f"final_checkpoints/toy_sequence/alphabet_{a}.ckpt",
            extra=(f"data.alphabet_size={a}", f"belief.alphabet_size={a}"),
            quick=("plots.0.num_samples=1024", "task.sampler.variable_noise.max_function_evals=20"),
        )
        for a in TOY_ALPHABETS
    ],
]

ROWS_BY_KEY = {row.key: row for row in ROWS}
TABLE12_KEYS = [row.key for row in ROWS if row.table in ("Table 1", "Table 2")]


#: Peak resident memory of one molecule row (Moses/GuacaMol). Generation alone
#: sits at ~12 GB (benchmark reference data plus the generated set) and the
#: metric phase adds a worker pool on top, so budget 16 GB per row.
MOLECULE_ROW_MEMORY_GB = 16


def _cgroup_memory_limits_gb() -> list[float]:
    """Every memory.max along this process' cgroup path.

    Under SLURM the limit sits on the job cgroup, several levels above the leaf
    the process itself lives in, so walk the whole chain.
    """
    limits: list[float] = []
    try:
        rel = Path("/proc/self/cgroup").read_text().strip().split(":")[-1]
    except OSError:
        return limits
    node = Path("/sys/fs/cgroup") / rel.lstrip("/")
    root = Path("/sys/fs/cgroup")
    while True:
        for name in ("memory.max", "memory/memory.limit_in_bytes"):
            try:
                raw = (node / name).read_text().strip()
            except OSError:
                continue
            if raw.isdigit():
                limits.append(int(raw) / 1024**3)
        if node == root or root not in node.parents:
            break
        node = node.parent
    return limits


def _total_memory_gb() -> float:
    """Memory available to this job, honouring a cgroup limit if there is one."""
    limits = _cgroup_memory_limits_gb()
    try:
        limits.append(
            os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3
        )
    except (ValueError, OSError):
        pass
    slurm_mem = os.environ.get("SLURM_MEM_PER_NODE")
    if slurm_mem and slurm_mem.isdigit():
        limits.append(int(slurm_mem) / 1024)
    return min(limits) if limits else 32.0


def max_molecule_jobs() -> int:
    """How many molecule rows can run at once without being OOM-killed."""
    return max(1, int(_total_memory_gb() * 0.85 // MOLECULE_ROW_MEMORY_GB))


def auto_generation_batch_size() -> int:
    """Pick a sampling batch size that keeps a large GPU busy.

    Molecules are sampled independently, so the batch size only changes how many
    of them are drawn per forward pass -- not the distribution.  The paper runs
    used 64, which leaves modern accelerators idle; the values below all divide
    both 15000 (GuacaMol) and 25000 (Moses) so the sample count is unchanged.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        mib = min(int(line) for line in out.split() if line.strip().isdigit())
    except Exception:
        return 64
    gib = mib / 1024
    if gib < 16:
        return 64
    if gib < 48:
        return 125
    return 250


def build_overrides(
    row: Row, *, quick: bool, extra: list[str], batch_size: int | None = None
) -> list[str]:
    ckpt = REPO_ROOT / row.checkpoint if row.checkpoint else None
    overrides = [
        f"experiment={row.experiment}",
        f"load_checkpoint={ckpt}",
        "wandb.mode=disabled",
        f"wandb.name=reproduce_{row.dataset}_{row.label}",
        f"task.sampler.variable_noise.max_function_evals={row.nfe}",
        f"task.sampler.variable_noise.step_method.method={row.method}",
        f"task.sampler.variable_noise.step_method.noise_scale={row.gamma}",
        "task.sampler.variable_noise.rho=1.0",
        "task.sampler.variable_noise.step_method.condition_on_prior=false",
    ]
    # Rows without a replayable paper run still get a fixed seed so that every
    # row of the reproduction is deterministic.
    overrides.append(f"seed={row.seed if row.seed is not None else 0}")
    if row.num_samples is not None:
        overrides.append(f"plots.0.num_samples={row.num_samples}")
    if batch_size is not None and row.batch_size_key is not None:
        overrides.append(f"{row.batch_size_key}={batch_size}")
    overrides.extend(row.extra)
    if quick:
        overrides.extend(row.quick)
    overrides.extend(extra)
    return overrides


def python_for(row: Row) -> str:
    """The interpreter of the environment the paper run used for this row."""
    path = ENV_PYTHON[row.env]
    if path.exists():
        return str(path)
    print(
        f"WARNING: {path} missing; running {row.key} with {sys.executable}. "
        f"Create the {row.env} environment for an exact reproduction.",
        flush=True,
    )
    return sys.executable


def gpu_name(device: str | None) -> str | None:
    query = ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]
    if device is not None:
        query.append(f"--id={device}")
    try:
        out = subprocess.run(query, capture_output=True, text=True, check=True).stdout
    except Exception:
        return None
    names = [line.strip() for line in out.splitlines() if line.strip()]
    return names[0] if names else None


def run_row(
    row: Row,
    *,
    quick: bool,
    extra: list[str],
    out_dir: Path,
    log_dir: Path,
    device: str | None,
    batch_size: int | None = None,
    cpus: int | None = None,
) -> dict[str, Any]:
    if row.checkpoint is not None:
        ckpt = REPO_ROOT / row.checkpoint
        if not ckpt.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {ckpt}\n"
                "Run `python scripts/download_checkpoints.py` first."
            )

    if row.batch_size_key is not None and row.default_batch_size is not None:
        batch_size = row.default_batch_size

    print(f"\n=== {row.key} ({row.table}) ===", flush=True)
    python = python_for(row)
    gpu = gpu_name(device)
    if row.gpu is not None and gpu is not None and row.gpu not in gpu:
        print(
            f"NOTE: {row.key} was produced on an {row.gpu}, this is an {gpu}; "
            "expect the right distribution but not the paper's exact draw.",
            flush=True,
        )
    started = time.time()
    log_path = log_dir / f"{row.dataset}_{row.label}.log"
    metrics: dict[str, float] = {}
    overrides: list[str] = []

    # Sampling batch sizes are a pure throughput knob, so an out-of-memory
    # failure is recoverable: halve and retry instead of losing the row.
    for attempt in range(4):
        overrides = build_overrides(row, quick=quick, extra=extra, batch_size=batch_size)
        print("Overrides:", " ".join(overrides), flush=True)

        with tempfile.TemporaryDirectory(prefix="graphbsi_metrics_") as tmp:
            metrics_path = Path(tmp) / "metrics.json"
            env = os.environ.copy()
            env["WANDB_MODE"] = "disabled"
            env["WANDB_SILENT"] = "true"
            env["GRAPHBSI_METRICS_JSON"] = str(metrics_path)
            env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
            if device is not None:
                env["CUDA_VISIBLE_DEVICES"] = device
            if cpus is not None:
                # molecule-benchmarks sizes its process pools from this; with
                # several rows in flight the default (all allocated cores) both
                # oversubscribes the CPUs and multiplies memory use.
                env["SLURM_CPUS_ON_NODE"] = str(cpus)
                env["OMP_NUM_THREADS"] = str(cpus)

            cmd = [python, str(REPO_ROOT / "train.py"), *overrides]
            with open(log_path, "w") as log_file:
                proc = subprocess.run(
                    cmd,
                    cwd=str(REPO_ROOT),
                    env=env,
                    check=False,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                )
            if proc.returncode == 0:
                if metrics_path.exists():
                    metrics = json.loads(metrics_path.read_text())
                break

        oom = "OutOfMemoryError" in log_path.read_text(errors="ignore")
        can_retry = (
            oom
            and row.batch_size_key is not None
            and batch_size is not None
            and batch_size > 1
            and attempt < 3
        )
        if not can_retry:
            raise RuntimeError(
                f"{row.key} failed with exit code {proc.returncode}; see {log_path}"
            )
        batch_size = max(1, batch_size // 2)
        print(f"{row.key} ran out of memory; retrying with batch size {batch_size}", flush=True)

    result = {
        "dataset": row.dataset,
        "label": row.label,
        "paper_table": row.table,
        "method": row.method,
        "gamma": row.gamma,
        "nfe": row.nfe,
        "checkpoint": row.checkpoint,
        "seed": row.seed,
        "source_run": row.source,
        "env": row.env,
        "paper_gpu": row.gpu,
        "gpu": gpu,
        "batch_size": batch_size,
        "overrides": [o.replace(f"{REPO_ROOT}/", "") for o in overrides],
        "metrics": metrics,
        "runtime_seconds": round(time.time() - started, 1),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "quick": quick,
    }
    out = out_dir / f"{row.dataset}_{row.label}.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"Wrote {out} ({result['runtime_seconds']:.0f}s)", flush=True)
    return result


def prepare_datasets(rows: list[Row], log_dir: Path, *, quick: bool = False) -> None:
    """Build the dataset and benchmark caches in a separate process.

    The first access to a dataset downloads and preprocesses it and builds the
    benchmark reference (which instantiates the FCD network). Doing that inside
    a row run would consume random numbers after seeding, so the samples would
    no longer match the paper run, which found these caches already in place.
    Building them up front also keeps parallel workers from racing on the same
    cache directory.
    """
    seen: set[str] = set()
    for row in rows:
        if row.experiment in seen:
            continue
        seen.add(row.experiment)
        print(f"Preparing data for {row.experiment}", flush=True)
        # Rebuilding the benchmark reference only matters inside a seeded row.
        extra = [o for o in row.extra if "rebuild_benchmarker" not in o]
        overrides = [f"experiment={row.experiment}", "wandb.mode=disabled", *extra]
        if row.num_samples is not None:
            overrides.append(f"plots.0.num_samples={row.num_samples}")
        if quick:
            overrides.extend(row.quick)
        log_path = log_dir / f"prepare_{row.dataset}.log"
        env = os.environ.copy()
        env["WANDB_MODE"] = "disabled"
        env["CUDA_VISIBLE_DEVICES"] = ""
        with open(log_path, "w") as log_file:
            proc = subprocess.run(
                [sys.executable, str(REPO_ROOT / "scripts" / "prepare_data.py"), *overrides],
                cwd=str(REPO_ROOT),
                env=env,
                check=False,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
        if proc.returncode != 0:
            print(
                f"!! data preparation for {row.dataset} exited with "
                f"{proc.returncode}; see {log_path}",
                flush=True,
            )


def _visible_devices() -> list[str]:
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env:
        return [d.strip() for d in env.split(",") if d.strip()]
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        return [line.strip() for line in out.splitlines() if line.strip()]
    except Exception:
        return ["0"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", help="Restrict to these datasets")
    parser.add_argument("--rows", nargs="+", help="Restrict to these rows, e.g. moses:ou_500")
    parser.add_argument(
        "--tables",
        nargs="+",
        choices=["1", "2", "3", "6", "9"],
        help="Paper tables to reproduce (1 and 2 are the main results)",
    )
    parser.add_argument("--all", action="store_true", help="Reproduce every known row")
    parser.add_argument("--quick", action="store_true", help="Reduced sample counts")
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of rows to evaluate in parallel (one GPU each)",
    )
    parser.add_argument(
        "--max-molecule-jobs",
        type=int,
        default=None,
        help="Cap on concurrent Moses/GuacaMol rows (default: from the memory limit)",
    )
    parser.add_argument(
        "--gen-batch-size",
        default="64",
        help="Molecules sampled per forward pass. 64 is what the paper runs used and "
        "the only value that reproduces their exact draw; 'auto' scales it with GPU "
        "memory for throughput (same distribution, different samples)",
    )
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--log-dir", type=Path, default=REPO_ROOT / "logs" / "rows")
    parser.add_argument("--override", action="append", default=[], help="Extra Hydra override")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip rows whose result JSON already exists (resume a partial run)",
    )
    parser.add_argument(
        "--list", action="store_true", help="List the selected rows and exit"
    )
    args = parser.parse_args()

    table_names = {
        "1": "Table 1",
        "2": "Table 2",
        "3": "Table 3",
        "6": "Table 6",
        "9": "Figure 9",
    }

    if args.all:
        selected = list(ROWS)
    else:
        selected = []
        if args.tables:
            wanted = {table_names[t] for t in args.tables}
            selected += [r for r in ROWS if r.table in wanted]
        if args.datasets:
            selected += [r for r in ROWS if r.dataset in args.datasets]
        if args.rows:
            for key in args.rows:
                if key not in ROWS_BY_KEY:
                    parser.error(f"Unknown row {key!r}. Known: {sorted(ROWS_BY_KEY)}")
                selected.append(ROWS_BY_KEY[key])
        if not selected:
            selected = [r for r in ROWS if r.key in TABLE12_KEYS]

    # de-duplicate, keep declaration order
    seen: set[str] = set()
    rows = [r for r in selected if not (r.key in seen or seen.add(r.key))]

    if args.list:
        for r in rows:
            print(
                f"{r.key:28s} {r.table:9s} {r.method:19s} gamma={r.gamma:<7} nfe={r.nfe:<5}"
                f" env={r.env:<5} gpu={r.gpu or '-':12s} source={r.source or '-'}"
            )
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    if args.skip_existing:
        rows = [
            r
            for r in rows
            if not (args.out_dir / f"{r.dataset}_{r.label}.json").exists()
        ]

    devices = _visible_devices()
    jobs = max(1, min(args.jobs, len(devices) if devices else 1))
    batch_size = (
        auto_generation_batch_size()
        if str(args.gen_batch_size).lower() == "auto"
        else int(args.gen_batch_size)
    )
    print(f"Reproducing {len(rows)} row(s) with {jobs} parallel job(s) on GPUs {devices}")
    print(f"Molecule sampling batch size: {batch_size}")
    cpus_per_job = max(2, (len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 4)) // jobs)

    sbm_rows = [r for r in rows if r.dataset == "sbm"]
    if sbm_rows:
        probe = subprocess.run(
            [python_for(sbm_rows[0]), "-c", "import graph_tool"], capture_output=True
        )
        if probe.returncode != 0:
            print(
                "WARNING: graph-tool is missing, so the SBM validity check falls back\n"
                "         to a spectral-clustering approximation and will score far\n"
                "         below Table 2. Run `pixi install` for the SBM rows.",
                flush=True,
            )

    results: list[dict[str, Any]] = []
    failures: list[tuple[str, str]] = []

    if rows:
        prepare_datasets(rows, args.log_dir, quick=args.quick)

    if jobs == 1:
        for row in rows:
            try:
                results.append(
                    run_row(
                        row,
                        quick=args.quick,
                        extra=args.override,
                        out_dir=args.out_dir,
                        log_dir=args.log_dir,
                        device=None,
                        batch_size=batch_size,
                        cpus=cpus_per_job,
                    )
                )
            except Exception as exc:  # keep going, report at the end
                print(f"!! {row.key} failed: {exc}", flush=True)
                failures.append((row.key, str(exc)))
    else:
        work: queue.Queue[Row] = queue.Queue()
        for row in rows:
            work.put(row)
        lock = threading.Lock()
        # A molecule row peaks at ~14 GB of host memory, so run only as many of
        # them at once as the memory limit allows -- exceeding it gets the job
        # SIGKILLed by the OOM killer, not just slowed down.
        molecule_slots = min(jobs, args.max_molecule_jobs or max_molecule_jobs())
        molecule_sem = threading.Semaphore(max(1, molecule_slots))
        print(f"At most {max(1, molecule_slots)} molecule row(s) at a time")

        def worker(device: str) -> None:
            while True:
                try:
                    row = work.get_nowait()
                except queue.Empty:
                    return
                heavy = row.num_samples is not None
                if heavy:
                    molecule_sem.acquire()
                try:
                    res = run_row(
                        row,
                        quick=args.quick,
                        extra=args.override,
                        out_dir=args.out_dir,
                        log_dir=args.log_dir,
                        device=device,
                        batch_size=batch_size,
                        cpus=cpus_per_job,
                    )
                    with lock:
                        results.append(res)
                except Exception as exc:
                    print(f"!! {row.key} failed: {exc}", flush=True)
                    with lock:
                        failures.append((row.key, str(exc)))
                finally:
                    if heavy:
                        molecule_sem.release()
                    work.task_done()

        threads = [
            threading.Thread(target=worker, args=(devices[i % len(devices)],), daemon=True)
            for i in range(jobs)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    # A row can fail for reasons that only exist while its neighbours run --
    # host memory pressure above all -- so give every failure one more try on
    # its own once the parallel phase is over.
    if failures and rows:
        retry_keys = {key for key, _ in failures}
        failures = []
        print(f"\nRetrying {len(retry_keys)} failed row(s) sequentially", flush=True)
        for row in rows:
            if row.key not in retry_keys:
                continue
            try:
                results.append(
                    run_row(
                        row,
                        quick=args.quick,
                        extra=args.override,
                        out_dir=args.out_dir,
                        log_dir=args.log_dir,
                        device=devices[0] if jobs > 1 else None,
                        batch_size=batch_size,
                        cpus=cpus_per_job,
                    )
                )
            except Exception as exc:
                print(f"!! {row.key} failed again: {exc}", flush=True)
                failures.append((row.key, str(exc)))

    # Always summarize everything present on disk so partial/resumed runs work.
    write_reports(args.out_dir)

    if failures:
        print("\nFailed rows:")
        for key, exc in failures:
            print(f"  {key}: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
