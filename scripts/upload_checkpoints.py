#!/usr/bin/env python
"""Upload local final_checkpoints/ to Hugging Face Hub."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi, create_repo


DEFAULT_REPO = "peteole/graphbsi-checkpoints"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=DEFAULT_REPO, help="HF model repo id")
    parser.add_argument(
        "--src",
        type=Path,
        default=Path("final_checkpoints"),
        help="Local checkpoint directory to upload",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the Hub repo as private",
    )
    args = parser.parse_args()

    if not args.src.exists():
        raise FileNotFoundError(f"Missing checkpoint directory: {args.src}")

    create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)
    api = HfApi()
    print(f"Uploading {args.src.resolve()} -> {args.repo}")
    api.upload_folder(
        folder_path=str(args.src),
        repo_id=args.repo,
        repo_type="model",
    )
    print("Done.")


if __name__ == "__main__":
    main()
