#!/usr/bin/env bash
# Reproduce all GraphBSI paper metrics from the released checkpoints.
#
# The script is meant to run unattended in a fresh clone: it creates a Python
# environment if necessary, downloads the checkpoints and then evaluates every
# row of Tabs. 1 and 2 of the paper, writing results/metrics.md.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

QUICK=0
RESUME=0
JOBS="${GRAPHBSI_JOBS:-}"
EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --quick)
      QUICK=1
      shift
      ;;
    --jobs)
      JOBS="$2"
      shift 2
      ;;
    --datasets|--rows|--tables)
      flag="$1"
      shift
      EXTRA+=("$flag")
      while [[ $# -gt 0 && "$1" != --* ]]; do
        EXTRA+=("$1")
        shift
      done
      ;;
    --all)
      EXTRA+=(--all)
      shift
      ;;
    --resume)
      RESUME=1
      shift
      ;;
    -h|--help)
      cat <<'USAGE'
Usage: reproduce_all.sh [--quick] [--jobs N] [--all] [--resume]
                        [--tables 1 2 ...] [--datasets NAME ...] [--rows ds:label ...]

Creates .venv if missing, downloads checkpoints if missing and runs
scripts/reproduce.py.  Without a selection it reproduces Tabs. 1 and 2.
Set GRAPHBSI_PYTHON to use a specific interpreter.
USAGE
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

# --- Python environments -----------------------------------------------------
# The paper runs used two environments, and an exact reproduction needs both:
#   .venv (uv.lock, torch 2.7, Python 3.12)   Moses, GuacaMol, planar, tree
#   .pixi (pixi.lock, torch 2.8, Python 3.13) SBM, whose validity metric needs
#                                             graph-tool from conda-forge
# scripts/reproduce.py picks the right one per row.
UV_PYTHON="$ROOT/.venv/bin/python"
PIXI_PYTHON="$ROOT/.pixi/envs/default/bin/python"

if [[ ! -x "$UV_PYTHON" ]]; then
  command -v uv >/dev/null 2>&1 || { echo "uv is required (https://docs.astral.sh/uv/)" >&2; exit 1; }
  echo "Creating .venv from uv.lock..."
  (cd "$ROOT" && uv sync --frozen)
fi
if [[ ! -x "$PIXI_PYTHON" ]]; then
  if command -v pixi >/dev/null 2>&1; then
    echo "Creating the pixi environment from pixi.lock..."
    (cd "$ROOT" && pixi install --frozen)
  else
    echo "WARNING: pixi not found, so the SBM rows run without graph-tool and" >&2
    echo "         will not match Table 2 (https://pixi.sh)." >&2
  fi
fi

PYTHON="${GRAPHBSI_PYTHON:-$UV_PYTHON}"
echo "Using Python: $PYTHON"
"$PYTHON" -c "import torch, lightning, hydra" >/dev/null

# --- Checkpoints -------------------------------------------------------------
for f in moses/last.ckpt guacamol/last.ckpt guacamol/epoch_59.ckpt \
         planar/last.ckpt sbm/last.ckpt tree/last.ckpt; do
  if [[ ! -f "final_checkpoints/$f" ]]; then
    echo "final_checkpoints/$f missing; downloading from Hugging Face..."
    "$PYTHON" scripts/download_checkpoints.py
    break
  fi
done

# --- Run ---------------------------------------------------------------------
if [[ ${#EXTRA[@]} -gt 0 ]]; then
  ARGS=("${EXTRA[@]}")
else
  ARGS=(--tables 1 2)
fi
if [[ "$QUICK" -eq 1 ]]; then
  ARGS+=(--quick)
fi
if [[ "$RESUME" -eq 1 ]]; then
  ARGS+=(--skip-existing)
fi
if [[ -z "$JOBS" ]]; then
  JOBS="$("$PYTHON" - <<'PY'
import subprocess
try:
    out = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                         capture_output=True, text=True, check=True).stdout
    print(max(1, len([l for l in out.splitlines() if l.strip()])))
except Exception:
    print(1)
PY
)"
fi
ARGS+=(--jobs "$JOBS")

mkdir -p logs
echo "Running: $PYTHON scripts/reproduce.py ${ARGS[*]}"
"$PYTHON" scripts/reproduce.py "${ARGS[@]}"
