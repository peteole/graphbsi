#!/usr/bin/env python
"""Download GraphBSI paper checkpoints from Hugging Face Hub."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

DEFAULT_REPO = "peteole/graphbsi-checkpoints"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=DEFAULT_REPO, help="HF model repo id")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("final_checkpoints"),
        help="Local destination directory",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional git revision / tag on the Hub",
    )
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {args.repo} -> {args.out.resolve()}")
    snapshot_download(
        repo_id=args.repo,
        repo_type="model",
        local_dir=str(args.out),
        revision=args.revision,
    )
    print("Done.")


if __name__ == "__main__":
    main()
