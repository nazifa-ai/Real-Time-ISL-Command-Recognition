#!/usr/bin/env python3
"""
Dataset acquisition for INCLUDE-50.

There are two official ways to obtain the data, and this script handles both:

``--keypoints`` (default, recommended: ~620 MB)
    The AI4Bharat pre-extracted MediaPipe keypoints
    (``https://zenodo.org/record/6674324/files/INCLUDE.zip``), which contain a
    ``Pose_Signs.zip`` archive with one ``.pkl`` per video:
    ``{keypoints: (T,75,3), confidences: (T,75), vid_shape: (h,w)}``.
    This is what the project trains on by default (``KEYPOINT_SOURCE=ai4bharat``).

``--videos`` (57 GB - usually not needed)
    The raw INCLUDE video release (Zenodo record 4010759, 46 zip parts).  Needed
    only if you want to re-extract landmarks yourself
    (``KEYPOINT_SOURCE=mediapipe``) or inspect video properties (FPS, duration).
    Run ``--videos --list`` to print the parts without downloading.

Operations
----------
    python training/fetch_dataset.py --keypoints              # download + verify
    python training/fetch_dataset.py --keypoints --extract    # also unzip to disk
    python training/fetch_dataset.py --keypoints --verify     # CRC check only
    python training/fetch_dataset.py --videos --list          # show part URLs

Downloads are resumable; ``--verify`` recomputes the archive CRC and reports
exactly which files are damaged, so a partial download can never be mistaken for
a complete one.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402

KEYPOINTS_URL = "https://zenodo.org/record/6674324/files/INCLUDE.zip?download=1"
KEYPOINTS_ARCHIVE = "INCLUDE.zip"
INNER_ARCHIVE = "INCLUDE/Pose_Signs.zip"
RAW_RECORD_API = "https://zenodo.org/api/records/4010759"


def _download(url: str, dest: Path, resume: bool = True, chunk_mb: int = 4) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    pos = dest.stat().st_size if (resume and dest.exists()) else 0
    headers = {"User-Agent": "isl-recognition/1.0"}
    if pos:
        headers["Range"] = f"bytes={pos}-"
        print(f"  resuming at {pos/1e6:.1f} MB")
    req = urllib.request.Request(url, headers=headers)
    mode = "ab" if pos else "wb"
    with urllib.request.urlopen(req, timeout=180) as r, open(dest, mode) as f:
        total = int(r.headers.get("Content-Length", 0)) + pos
        done = pos
        while True:
            chunk = r.read(chunk_mb << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done/1e6:7.1f} / {total/1e6:7.1f} MB "
                      f"({100*done/total:5.1f}%)", end="", flush=True)
    print()
    return dest


def extract_inner_archive(outer: Path, out_dir: Path, force: bool = False) -> Path:
    """Extract ``INCLUDE/Pose_Signs.zip`` from the outer archive.

    The outer member is *stored* (uncompressed), so if its CRC is broken the bytes
    can still be recovered by copying them verbatim - this keeps a partially
    damaged download usable instead of failing outright.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "Pose_Signs.zip"
    if target.exists() and not force:
        print(f"  exists  {target}")
        return target
    z = zipfile.ZipFile(outer)
    info = z.getinfo(INNER_ARCHIVE)
    with open(outer, "rb") as f:
        f.seek(info.header_offset)
        sig, ver, flag, comp, mt, md, crc, csize, usize, nlen, elen = struct.unpack(
            "<IHHHHHIIIHH", f.read(30))
        assert sig == 0x04034B50, "bad local header"
        f.seek(info.header_offset + 30 + nlen + elen)
        remaining = info.compress_size
        print(f"  copying {info.compress_size/1e6:.1f} MB (stored member) -> {target}")
        with open(target, "wb") as out:
            while remaining > 0:
                chunk = f.read(min(8 << 20, remaining))
                if not chunk:
                    break
                out.write(chunk)
                remaining -= len(chunk)
    return target


def verify_archive(path: Path, write_json: Optional[Path] = None) -> Dict[str, object]:
    """Per-entry CRC check; reports every damaged file (not just the first)."""
    z = zipfile.ZipFile(path)
    names = [n for n in z.namelist() if not n.endswith("/")]
    bad: List[str] = []
    for i, n in enumerate(names):
        try:
            with z.open(n) as f:
                while f.read(8 << 20):
                    pass
        except Exception as exc:
            bad.append(f"{n}: {type(exc).__name__}: {str(exc)[:60]}")
        if (i + 1) % 500 == 0:
            print(f"\r  verified {i+1}/{len(names)} (damaged: {len(bad)})", end="", flush=True)
    print()
    report = {"archive": str(path), "entries": len(names), "damaged": bad}
    if write_json:
        Path(write_json).write_text(json.dumps(report, indent=2))
    return report


def extract_keypoints(archive: Path, dest: Path, limit: Optional[int] = None) -> int:
    """Extract readable ``.pkl`` entries into a folder tree (skipping damaged ones)."""
    dest.mkdir(parents=True, exist_ok=True)
    z = zipfile.ZipFile(archive)
    names = [n for n in z.namelist() if n.endswith(".pkl")]
    if limit:
        names = names[:limit]
    ok = skipped = 0
    for i, n in enumerate(names):
        out = dest / Path(n).relative_to("Pose_Signs") if n.startswith("Pose_Signs/") else dest / Path(n).name
        if out.exists():
            ok += 1
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            with z.open(n) as src, open(out, "wb") as dst:
                while True:
                    b = src.read(4 << 20)
                    if not b:
                        break
                    dst.write(b)
            ok += 1
        except Exception:
            skipped += 1
            if out.exists():
                out.unlink()
        if (i + 1) % 500 == 0:
            print(f"\r  extracted {i+1}/{len(names)} (skipped {skipped})", end="", flush=True)
    print()
    print(f"  extracted {ok} files, skipped {skipped} damaged")
    return ok


def list_raw_video_parts() -> List[Dict[str, object]]:
    req = urllib.request.Request(RAW_RECORD_API, headers={"User-Agent": "isl-recognition/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        rec = json.loads(r.read())
    return [{"key": f["key"], "size_mb": round(f["size"] / 1e6, 1),
             "url": f["links"]["self"]} for f in rec.get("files", [])]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--keypoints", action="store_true", help="download the pre-extracted keypoints (default action)")
    ap.add_argument("--videos", action="store_true", help="download/inspect the 57 GB raw video release")
    ap.add_argument("--list", action="store_true", help="only list what would be downloaded")
    ap.add_argument("--extract", action="store_true", help="extract .pkl files to disk after download")
    ap.add_argument("--verify", action="store_true", help="CRC-check an existing download")
    ap.add_argument("--out-dir", default="/data/include", help="where archives are kept")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    do_keypoints = args.keypoints or not args.videos

    if args.videos:
        parts = list_raw_video_parts()
        total = sum(p["size_mb"] for p in parts)
        print(f"Raw INCLUDE videos: {len(parts)} parts, {total/1000:.1f} GB total")
        for p in parts[:10]:
            print(f"  {p['key']:35s} {p['size_mb']:8.1f} MB")
        print("  ...")
        print("\nThese are NOT downloaded automatically (57 GB). Fetch a part with:")
        print(f"  curl -L -C - -o {out_dir}/<part>.zip '<url>'")
        print("then unzip into", config.DATASET_PATH)
        return 0

    if not do_keypoints:  # pragma: no cover
        return 0

    archive = out_dir / KEYPOINTS_ARCHIVE
    if args.list:
        print(f"keypoints: {KEYPOINTS_URL}\n  -> {archive}")
        return 0

    if not (archive.exists() and archive.stat().st_size > 0):
        print(f"Downloading INCLUDE keypoints (~620 MB)\n  {KEYPOINTS_URL}\n  -> {archive}")
        _download(KEYPOINTS_URL, archive)

    print("Verifying outer archive ...")
    try:
        report = verify_archive(archive, write_json=out_dir / "corrupt_scan.json")
        print(f"  outer archive OK ({report['entries']} entries)")
    except zipfile.BadZipFile:
        print("  outer archive is damaged; copying the stored inner member verbatim ...")
        inner = extract_inner_archive(archive, out_dir, force=args.force)
        report = verify_archive(inner, write_json=out_dir / "corrupt_scan.json")
    damaged = report.get("damaged", [])
    print(f"  entries: {report['entries']}, damaged: {len(damaged)}")
    if damaged:
        print("  damaged entries are reported in corrupt_scan.json and skipped by the loaders.")

    inner = extract_inner_archive(archive, out_dir, force=False) if not (out_dir / "Pose_Signs.zip").exists() else out_dir / "Pose_Signs.zip"

    if args.extract:
        dest = Path(config.KEYPOINTS_PATH)
        print(f"Extracting keypoints -> {dest}")
        extract_keypoints(inner, dest)

    from backend.preprocessing import include50

    index, summary = include50.build_index()

    print(f"\nINCLUDE-50 index: {len(index)} videos, {index['label'].nunique()} classes "
      f"(train={summary.train}, val={summary.val}, test={summary.test})")

    print("\nDataset archive is ready.")
    print("Next:  python training/audit_dataset.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
