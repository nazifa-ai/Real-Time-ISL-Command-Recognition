#!/usr/bin/env python3
"""
PHASE 12 -- batch / offline inference (video upload and dataset sweep).

Examples
--------
    # one uploaded video file
    python backend/inference/batch.py --video uploads/hello.mp4

    # a whole folder
    python backend/inference/batch.py --video-dir uploads/ --recursive

    # 40 test clips of the session-disjoint split, with ground truth
    python backend/inference/batch.py --from-dataset --split test --limit 40

    # every clip of one class
    python backend/inference/batch.py --from-dataset --class HELLO --split test

Outputs ``results/batch_predictions_<model>.csv`` with one row per clip:
file, true_class, predicted_class, confidence, status, latency_ms, correct.
When ground truth is available a top-1 accuracy over that run is printed *and*
written to ``results/batch_summary_<model>.json``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import config  # noqa: E402
from backend.inference import audit  # noqa: E402
from backend.inference.frames import dataset_features, features_from_video  # noqa: E402
from backend.inference.predictor import PredictionEngine  # noqa: E402


# --------------------------------------------------------------------------
# sources
# --------------------------------------------------------------------------
def clips_from_dataset(split: str = "test", protocol: Optional[str] = None,
                       class_name: Optional[str] = None, limit: Optional[int] = None) -> List[Dict[str, str]]:
    from backend.preprocessing import include50

    rows = include50.load_index(protocol).to_dict("records")
    if split and split != "all":
        rows = [r for r in rows if str(r.get("split")) == split]
    if class_name:
        target = class_name.strip().upper()
        rows = [r for r in rows if str(r.get("label", "")).strip().upper() == target]
    if limit:
        rows = rows[: int(limit)]
    return [{"path": r["FilePath"], "true_class": str(r.get("label"))} for r in rows]


def clips_from_videos(path: Path, recursive: bool, limit: Optional[int]) -> List[Dict[str, str]]:
    exts = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
    pattern = "**/*" if recursive else "*"
    files = sorted(p for p in path.glob(pattern) if p.suffix.lower() in exts and p.is_file())
    if limit:
        files = files[: int(limit)]
    stem_to_class = {c.upper(): c for c in _vocabulary()}
    out = []
    for f in files:
        # a filename like "HELLO_clip3.mp4" carries its (optional) ground truth
        guessed = f.stem.split("_")[0].upper() if "_" in f.stem else ""
        out.append({"path": str(f), "true_class": stem_to_class.get(guessed, "")})
    return out


def _vocabulary() -> List[str]:
    from backend.preprocessing.include50 import load_vocabulary

    return load_vocabulary()


# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------
def run(clips: Iterable[Dict[str, str]], engine: PredictionEngine, features_loader=None,
        log: bool = False, source: str = "batch") -> Dict[str, object]:
    features_loader = features_loader or (lambda p: dataset_features(p))
    rows: List[Dict[str, object]] = []
    latencies: List[float] = []
    t_start = time.time()

    for item in clips:
        path = item["path"]
        true_class = (item.get("true_class") or "").strip()
        try:
            feats = features_loader(path)
        except Exception as exc:  # unreadable file must not kill the sweep
            rows.append({"file": Path(path).name, "true_class": true_class,
                         "predicted_class": "ERROR", "confidence": 0.0,
                         "status": "ERROR", "latency_ms": 0.0, "correct": "",
                         "error": f"{type(exc).__name__}: {exc}"})
            continue
        if feats is None:
            rows.append({"file": Path(path).name, "true_class": true_class,
                         "predicted_class": "SKIPPED", "confidence": 0.0,
                         "status": "SKIPPED", "latency_ms": 0.0, "correct": "",
                         "error": "no usable landmarks"})
            continue

        engine.clear_window()              # independent clips must not share smoothing state
        # one forward pass per clip: score once, then derive the clip-level decision and
        # the top-3.  (Calling predict() as well would double the measured latency.)
        t0 = time.perf_counter()
        probs = engine.probabilities(feats)
        idx = int(np.argmax(probs))
        raw_conf = float(probs[idx])
        latency_ms = (time.perf_counter() - t0) * 1000.0
        raw_sign = engine.classes[idx] if idx < len(engine.classes) else str(idx)
        top3 = np.argsort(probs)[::-1][:3]
        clipped_sign = raw_sign if raw_conf >= engine.threshold else config.UNCERTAIN_LABEL
        row = {
            "file": Path(path).name,
            "true_class": true_class,
            "predicted_class": raw_sign,                 # clip-level argmax
            "confidence": round(raw_conf, 6),
            "status": "CONFIDENT" if raw_conf >= engine.threshold else config.UNCERTAIN_LABEL,
            "gated_sign": clipped_sign,
            "latency_ms": round(latency_ms, 3),
            "correct": ("" if not true_class else int(raw_sign == true_class)),
            "top3": "|".join(f"{engine.classes[i]}:{probs[i]:.4f}" for i in top3),
            "error": "",
        }
        rows.append(row)
        latencies.append(latency_ms)
        if log:
            audit.log_prediction(clipped_sign, raw_conf, row["status"], latency_ms,
                                 engine.model_name, source=source,
                                 extra={"file": row["file"], "raw_sign": raw_sign})
        print(f"{row['file']:<28} pred={row['predicted_class']:<22} conf={row['confidence']:.3f} "
              f"{(('true=' + true_class) if true_class else ''):<24} "
              f"{'OK' if row['correct'] == 1 else ('x' if row['correct'] == 0 else '')}")

    labelled = [r for r in rows if r["correct"] != ""]
    n_correct = sum(int(r["correct"]) for r in labelled)
    mean_lat = float(np.mean(latencies)) if latencies else 0.0
    summary = {
        "model": engine.model_name,
        "clips": len(rows),
        "labelled_clips": len(labelled),
        "accuracy": (n_correct / len(labelled)) if labelled else None,
        "uncertain_rate": (sum(1 for r in rows if r["status"] == config.UNCERTAIN_LABEL) / len(rows)) if rows else 0.0,
        "mean_latency_ms": round(mean_lat, 3),
        "p95_latency_ms": round(float(np.percentile(latencies, 95)), 3) if latencies else 0.0,
        "throughput_fps": round(1000.0 / mean_lat, 2) if mean_lat else 0.0,
        "wall_seconds": round(time.time() - t_start, 2),
        "errors": sum(1 for r in rows if r["status"] == "ERROR"),
        "skipped": sum(1 for r in rows if r["status"] == "SKIPPED"),
    }
    return {"rows": rows, "summary": summary}


def write_outputs(result: Dict[str, object], out_dir: Optional[Path] = None,
                  model_name: str = "") -> Dict[str, Path]:
    out_dir = Path(out_dir or config.RESULTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = result["rows"]  # type: ignore[assignment]
    csv_path = out_dir / f"batch_predictions_{model_name}.csv"
    json_path = out_dir / f"batch_summary_{model_name}.json"
    if rows:
        fields = list(dict.fromkeys(k for r in rows for k in r.keys()))
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=fields)
            wr.writeheader()
            for r in rows:
                wr.writerow(r)
    json_path.write_text(json.dumps(result["summary"], indent=2))
    return {"csv": csv_path, "json": json_path}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", help="single video file")
    src.add_argument("--video-dir", help="folder of videos")
    src.add_argument("--from-dataset", action="store_true", help="run over INCLUDE keypoint clips")
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--split", default="test")
    ap.add_argument("--protocol", default=None, choices=("session_disjoint", "official"))
    ap.add_argument("--class", dest="class_name", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--model", default=config.DEFAULT_MODEL, choices=list(config.AVAILABLE_MODELS))
    ap.add_argument("--threshold", type=float, default=config.CONFIDENCE_THRESHOLD)
    ap.add_argument("--log", action="store_true", help="also append events to the JSONL audit log")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args(argv)

    try:
        engine = PredictionEngine(model_name=args.model, threshold=args.threshold)
    except FileNotFoundError as exc:
        print(f"error: {exc}")
        return 2

    if args.from_dataset:
        clips = clips_from_dataset(args.split, args.protocol, args.class_name, args.limit)
        loader = dataset_features
    elif args.video:
        clips = [{"path": args.video, "true_class": ""}]
        loader = None
    else:
        clips = clips_from_videos(Path(args.video_dir), args.recursive, args.limit)
        loader = None

    if not clips:
        print("no clips matched the requested source/filters")
        return 1

    print(f"model={args.model} threshold={args.threshold} clips={len(clips)}")
    result = run(clips, engine, features_loader=loader, log=args.log)
    paths = write_outputs(result, args.out_dir, args.model)
    print("\nSummary")
    for k, v in result["summary"].items():
        print(f"  {k}: {v}")
    print(f"\nwrote {paths['csv']}\nwrote {paths['json']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
