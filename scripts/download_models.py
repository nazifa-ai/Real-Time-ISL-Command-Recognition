#!/usr/bin/env python3
"""
Download the MediaPipe Tasks model bundles used by the extractor.

MediaPipe >= 0.10.18 (and 1.x, which this project runs on) no longer ships the
legacy ``mediapipe.solutions`` API; landmark extraction therefore uses the Tasks
API with explicit ``.task`` bundles.

Usage
-----
    python scripts/download_models.py            # holistic (default) + pose + hand
    python scripts/download_models.py --only holistic
    python scripts/download_models.py --check    # verify what is already present
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402

ASSETS_DIR = Path(config.MODELS_DIR) / "mediapipe_assets"

BUNDLES = {
    "holistic": "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/"
                "holistic_landmarker/float16/1/holistic_landmarker.task",
    "pose": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
            "pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    "hand": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
            "hand_landmarker/float16/1/hand_landmarker.task",
    "pose_full": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
                 "pose_landmarker_full/float16/1/pose_landmarker_full.task",
}
MIN_BYTES = 1_000_000


def fetch(name: str, url: str, force: bool = False) -> Path:
    dest = ASSETS_DIR / Path(url).name
    if dest.exists() and dest.stat().st_size > MIN_BYTES and not force:
        print(f"  present {dest.name} ({dest.stat().st_size/1e6:.1f} MB)")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  fetch   {dest.name} <- {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "isl-recognition/1.0"})
    with urllib.request.urlopen(req, timeout=300) as r, open(dest, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    size = dest.stat().st_size
    if size < MIN_BYTES:
        raise RuntimeError(f"{dest.name} looks truncated ({size} bytes)")
    print(f"  saved   {dest.name} ({size/1e6:.1f} MB)")
    return dest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=list(BUNDLES), default=None)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    targets = {args.only: BUNDLES[args.only]} if args.only else \
        {k: v for k, v in BUNDLES.items() if k != "pose_full"}

    print(f"MediaPipe assets -> {ASSETS_DIR}")
    missing = []
    for name, url in targets.items():
        path = ASSETS_DIR / Path(url).name
        if args.check:
            ok = path.exists() and path.stat().st_size > MIN_BYTES
            print(f"  {'OK     ' if ok else 'MISSING'} {path.name}")
            if not ok:
                missing.append(name)
        else:
            fetch(name, url, force=args.force)

    if args.check and missing:
        print(f"\nRun `python scripts/download_models.py --only {missing[0]}` to fetch the missing bundle(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
