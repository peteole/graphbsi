#!/usr/bin/env python
"""Download / prepare datasets needed to reproduce GraphBSI paper metrics.

Molecular (Moses, GuacaMol, QM9) and synthetic-graph (planar, SBM, tree) data
are auto-downloaded by molecule-benchmarks / synthetic-graph-benchmarks on the
first training or evaluation run. This script only fetches the DNA enhancer
assets that are not auto-downloaded.
"""

from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path
from urllib.request import urlopen, urlretrieve

ZENODO_RECORD = "10184648"
WORKDIR_URL = "https://publbuck.s3.us-east-2.amazonaws.com/workdir.zip"
DNA_PICKLES = ("DeepFlyBrain_data.pkl", "DeepMEL2_data.pkl")


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"Already present: {dest}")
        return
    print(f"Downloading {url} -> {dest}")
    urlretrieve(url, dest)


def download_dna_pickles(cache_dir: Path) -> None:
    """Fetch enhancer pickles from the Dirichlet Flow Matching Zenodo record."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    missing = [name for name in DNA_PICKLES if not (cache_dir / name).exists()]
    if not missing:
        print(f"DNA pickles already present in {cache_dir}")
        return

    api_url = f"https://zenodo.org/api/records/{ZENODO_RECORD}"
    print(f"Querying Zenodo record {ZENODO_RECORD}...")
    with urlopen(api_url, timeout=60) as resp:
        meta = json.loads(resp.read().decode("utf-8"))
    files = {f["key"]: f["links"]["self"] for f in meta.get("files", [])}

    archive_keys = [k for k in files if k.endswith(".zip") or "the_code" in k.lower()]
    if archive_keys:
        key = archive_keys[0]
        archive_path = cache_dir / key
        _download(files[key], archive_path)
        print(f"Extracting {archive_path}...")
        with zipfile.ZipFile(archive_path) as zf:
            members = zf.namelist()
            for pickle_name in DNA_PICKLES:
                matches = [m for m in members if m.endswith(pickle_name)]
                if not matches:
                    raise FileNotFoundError(
                        f"{pickle_name} not found in {archive_path}"
                    )
                src = matches[0]
                dest = cache_dir / pickle_name
                with zf.open(src) as src_f, open(dest, "wb") as dst_f:
                    shutil.copyfileobj(src_f, dst_f)
                print(f"Wrote {dest}")
        return

    for pickle_name in missing:
        if pickle_name not in files:
            raise FileNotFoundError(
                f"Could not find {pickle_name} in Zenodo record {ZENODO_RECORD}. "
                "Download manually from https://zenodo.org/records/10184648 "
                f"and place it in {cache_dir}/"
            )
        _download(files[pickle_name], cache_dir / pickle_name)


def download_classifier_workdir(workdir: Path) -> None:
    """Download pretrained FBD classifier checkpoints (workdir.zip)."""
    workdir.mkdir(parents=True, exist_ok=True)
    marker = workdir / "clsDNAclean_cnn_1stack_2023-12-30_15-01-30"
    if marker.exists():
        print(f"Classifier workdir already present: {workdir}")
        return

    zip_path = workdir.parent / "workdir.zip"
    _download(WORKDIR_URL, zip_path)
    print(f"Extracting {zip_path} -> {workdir}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(workdir.parent)
    extracted = workdir.parent / "workdir"
    if extracted.exists() and extracted.resolve() != workdir.resolve():
        for child in extracted.iterdir():
            target = workdir / child.name
            if not target.exists():
                shutil.move(str(child), str(target))
        shutil.rmtree(extracted, ignore_errors=True)
    print(f"Classifier workdir ready at {workdir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dna",
        action="store_true",
        help="Download DNA enhancer pickles + FBD classifier workdir",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all optional assets (currently DNA only)",
    )
    parser.add_argument(
        "--dna-cache-dir",
        type=Path,
        default=Path("data/dna"),
        help="Where to store DeepFlyBrain / DeepMEL2 pickles",
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=Path("data/dna/workdir"),
        help="Where to extract FBD classifier checkpoints",
    )
    args = parser.parse_args()

    if not (args.dna or args.all):
        parser.print_help()
        print(
            "\nNote: Moses / GuacaMol / QM9 / planar / SBM / tree download "
            "automatically on first run. Pass --dna or --all for enhancer assets."
        )
        return

    download_dna_pickles(args.dna_cache_dir)
    download_classifier_workdir(args.workdir)
    print("Done.")


if __name__ == "__main__":
    main()
