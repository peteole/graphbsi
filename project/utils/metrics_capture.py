"""Capture scalar metrics during validation for offline reproduction."""

from __future__ import annotations

import json
import numbers
import os
from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import wandb

_CAPTURED: dict[str, float] = {}
_PATH: Path | None = None


def _to_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, numbers.Real):
        return float(value)
    if hasattr(value, "item"):
        try:
            item = value.item()
        except Exception:
            return None
        if isinstance(item, numbers.Real) and not isinstance(item, bool):
            return float(item)
    return None


def record(key: str, value: Any) -> None:
    number = _to_float(value)
    if number is None:
        return
    _CAPTURED[key] = number
    flush()


def record_many(data: dict[str, Any]) -> None:
    for key, value in data.items():
        number = _to_float(value)
        if number is not None:
            _CAPTURED[key] = number
    if data:
        flush()


def flush() -> None:
    if _PATH is None:
        return
    _PATH.parent.mkdir(parents=True, exist_ok=True)
    _PATH.write_text(json.dumps(_CAPTURED, indent=2))


def install_wandb_hook() -> None:
    """Patch wandb.log so scalar metrics are captured even after wandb.init."""
    global _PATH
    metrics_path = os.environ.get("GRAPHBSI_METRICS_JSON")
    if not metrics_path:
        return
    _PATH = Path(metrics_path)

    # Avoid stacking wrappers if called repeatedly.
    if getattr(wandb.log, "_graphbsi_capture", False):
        return

    original_log = wandb.log

    def _log(data=None, *args, **kwargs):
        if isinstance(data, dict):
            record_many(data)
        try:
            return original_log(data, *args, **kwargs)
        except Exception:
            return None

    _log._graphbsi_capture = True  # type: ignore[attr-defined]
    wandb.log = _log

    import atexit

    atexit.register(flush)


class MetricsCaptureCallback(pl.Callback):
    """Reinstall the wandb hook around validation (WandbLogger may replace it)."""

    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str) -> None:
        install_wandb_hook()

    def on_validation_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        install_wandb_hook()

    def on_validation_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        install_wandb_hook()

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        flush()

    def teardown(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str
    ) -> None:
        flush()
