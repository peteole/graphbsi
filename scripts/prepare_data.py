#!/usr/bin/env python
"""Materialize a dataset cache without running an evaluation.

``scripts/reproduce.py`` calls this once per dataset before it evaluates any
row. The first access to a dataset downloads and preprocesses it and builds the
benchmark reference; doing that inside a seeded row run would shift the random
state and change the generated samples.

Usage:
  python scripts/prepare_data.py experiment=moses/reproduce
"""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from project.utils import set_seed  # noqa: E402


@hydra.main(config_path="../config", config_name="train", version_base=None)
def main(config: DictConfig) -> None:
    OmegaConf.resolve(config)
    rng = set_seed(config)
    datamodule = instantiate(config.data, rng=rng)
    datamodule.prepare_data()
    datamodule.setup("fit")
    # The class conditioner and the belief cache statistics of the training set.
    if config.get("class_conditioner") is not None:
        instantiate(config.class_conditioner, datamodule=datamodule)
    instantiate(config.belief, datamodule=datamodule)
    print(f"Prepared data for {config.data.get('ds_type', config.data)}")

    # Instantiating the plot callbacks builds and caches the benchmark reference
    # statistics (e.g. the FCD reference for a given sample count). Doing that
    # here keeps parallel workers from writing the same cache file at once.
    for cb_config in config.get("plots", []) or []:
        try:
            instantiate(cb_config, datamodule=datamodule)
        except Exception as exc:  # best effort -- the row run will redo it
            print(f"Skipping callback warm-up ({exc})")


if __name__ == "__main__":
    main()
