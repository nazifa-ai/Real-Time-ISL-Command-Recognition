#!/usr/bin/env python3
"""
Download the official INCLUDE metadata into ``datasets/metadata/``.

Two protocols are supported and both are *official* AI4Bharat sources:

``--protocol official``  (default)
    AI4Bharat OpenHands split files - these are the files used throughout this
    project because they define the official INCLUDE / INCLUDE-50 train/test split:
        assets/include_metadata/Train_Test_Split/{train,test}_include{50,}.csv
        assets/include_metadata/normalized_glosses.csv

``--protocol hf``
    HuggingFace ``ai4bharat/INCLUDE`` parquet files, which additionally ship a
    validation split (used when ``SPLIT_PROTOCOL=hf``).

Usage
-----
    python training/fetch_metadata.py
    python training/fetch_metadata.py --protocol hf
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402

OH_BASE = "https://raw.githubusercontent.com/AI4Bharat/OpenHands/main/openhands/datasets/assets/include_metadata"
HF_BASE = "https://huggingface.co/api/datasets/ai4bharat/INCLUDE/parquet/default"

OFFICIAL_FILES = {
    "train_include50.csv": f"{OH_BASE}/Train_Test_Split/train_include50.csv",
    "test_include50.csv": f"{OH_BASE}/Train_Test_Split/test_include50.csv",
    "train_include.csv": f"{OH_BASE}/Train_Test_Split/train_include.csv",
    "test_include.csv": f"{OH_BASE}/Train_Test_Split/test_include.csv",
    "normalized_glosses.csv": f"{OH_BASE}/normalized_glosses.csv",
}
HF_FILES = {
    "include_train.parquet": f"{HF_BASE}/train/0.parquet",
    "include_val.parquet": f"{HF_BASE}/val/0.parquet",
    "include_test.parquet": f"{HF_BASE}/test/0.parquet",
}


def download(url: str, dest: Path, force: bool = False) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force and dest.stat().st_size > 0:
        print(f"  exists  {dest.name} ({dest.stat().st_size} bytes)")
        return dest
    print(f"  fetch   {dest.name}  <- {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "isl-recognition/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
        f.write(r.read())
    print(f"  saved   {dest.name} ({dest.stat().st_size} bytes)")
    return dest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--protocol", default="official", choices=["official", "hf", "all"])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    files = {}
    if args.protocol in ("official", "all"):
        files.update(OFFICIAL_FILES)
    if args.protocol in ("hf", "all"):
        files.update(HF_FILES)

    print(f"Downloading INCLUDE metadata into {config.METADATA_DIR}")
    for name, url in files.items():
        download(url, Path(config.METADATA_DIR) / name, force=args.force)

    # sanity: the vocabulary must be derivable from what we just fetched
    from backend.preprocessing import include50

    vocab = include50.derive_vocabulary("official")
    out = include50.save_vocabulary("official")
    print(f"\nDerived vocabulary: {len(vocab)} classes -> {out}")
    print("   " + ", ".join(vocab[:12]) + (" ..." if len(vocab) > 12 else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
