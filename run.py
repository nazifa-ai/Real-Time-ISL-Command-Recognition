#!/usr/bin/env python3
"""
Single entry point for every part of the project.

    python run.py info                     show configuration + what exists on disk
    python run.py audit                    PHASE 2  dataset audit  -> docs/dataset_audit.md
    python run.py metadata                 verify/download the official INCLUDE-50 metadata
    python run.py preprocess               PHASE 4  build sequence tensors
    python run.py train [--model gru_mha]  PHASES 5-7 train LSTM / GRU / GRU+MHA
    python run.py evaluate                 PHASE 8  per-class comparison + summary metrics
    python run.py robustness               PHASE 9  perturbation experiments
    python run.py threshold                PHASE 9b confidence-threshold sweep
    python run.py realtime                 PHASE 12 webcam recognition (local machine)
    python run.py batch --from-dataset     PHASE 12 offline inference over clips
    python run.py api                      PHASE 13 FastAPI service
    python run.py test [--args ...]        PHASE 16 pytest suite
    python run.py doctor                   environment / asset self-check

Every command forwards extra flags to the underlying script, e.g.

    python run.py train --model gru --epochs 40
    python run.py test -k api -q
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402


def _run(script: str, extra: list) -> int:
    cmd = [sys.executable, str(ROOT / script), *extra]
    print(f"$ {' '.join(cmd)}")
    return subprocess.call(cmd, cwd=str(ROOT))


def _run_module(module: str, extra: list) -> int:
    cmd = [sys.executable, "-m", module, *extra]
    print(f"$ {' '.join(cmd)}")
    return subprocess.call(cmd, cwd=str(ROOT))


def _keypoint_archive():
    """Locate the keypoint archive the same way the loader does."""
    try:
        from backend.preprocessing.include50 import keypoint_archive

        return keypoint_archive() or "not found (see docs/setup.md)"
    except Exception as exc:
        return f"unavailable: {exc}"


def cmd_info(_args) -> int:
    print(config.describe() if hasattr(config, "describe") else "")
    print("Configuration")
    print("=" * 72)
    for key in ("SPLIT_PROTOCOL", "SEQUENCE_LENGTH", "FEATURE_DIM", "MODEL_INPUT_DIM",
                "MASK_ENABLED", "NUM_CLASSES", "CONFIDENCE_THRESHOLD", "SMOOTHING_WINDOW",
                "STABLE_FRAMES_REQUIRED", "TTS_COOLDOWN", "DEFAULT_MODEL", "RANDOM_SEED",
                "KEYPOINT_SOURCE"):
        if hasattr(config, key):
            print(f"  {key:<22}: {getattr(config, key)}")

    print("\nArtefacts")
    print("=" * 72)
    checks = [
        ("official metadata", Path(config.METADATA_DIR) / config.METADATA_FILES["train_50"]),
        ("frozen vocabulary", Path(config.METADATA_DIR) / "vocabulary.json"),
        ("split index", Path(config.METADATA_DIR) / f"index_{config.SPLIT_PROTOCOL}.csv"),
        ("keypoint archive", _keypoint_archive()),
        ("processed tensors", Path(config.PROCESSED_DIR) / f"sequences_{config.SPLIT_PROTOCOL}.npz"),
    ]
    for name in config.AVAILABLE_MODELS:
        checks.append((f"model {name}", Path(config.MODEL_PATHS[name]) / "model.keras"))
    checks += [
        ("dataset audit", Path(config.DOCS_DIR) / "dataset_audit.md"),
        ("evaluation summary", Path(config.RESULTS_DIR) / "summary_metrics.csv"),
        ("robustness results", Path(config.RESULTS_DIR) / "robustness_results.csv"),
        ("audit log", Path(config.LOG_PATHS["predictions"])),
    ]
    for name, path in checks:
        mark = "OK   " if Path(path).exists() else "--   "
        print(f"  [{mark}] {name:<22} {path}")

    print("\nNext steps if something is missing:")
    print("  python run.py audit        # needs the INCLUDE keypoints (see docs/setup.md)")
    print("  python run.py preprocess && python run.py train && python run.py evaluate")
    return 0


def cmd_doctor(_args) -> int:
    print("Environment self-check")
    print("=" * 72)
    problems = []
    for mod in ("numpy", "pandas", "tensorflow", "mediapipe", "cv2", "fastapi",
                "pydantic", "requests", "matplotlib"):
        try:
            m = __import__(mod)
            print(f"  OK    {mod} {getattr(m, '__version__', '')}")
        except Exception as exc:
            problems.append(f"{mod}: {exc}")
            print(f"  FAIL  {mod}: {exc}")

    print("\nAssets")
    print("=" * 72)
    assets = Path(config.MODELS_DIR) / "mediapipe_assets"
    bundles = list(assets.glob("*.task")) if assets.exists() else []
    print(f"  MediaPipe .task bundles: {len(bundles)} in {assets}")
    if not bundles:
        problems.append("no MediaPipe .task bundles - run: python scripts/download_models.py")

    try:
        from backend.preprocessing.include50 import keypoint_archive, load_vocabulary

        print(f"  keypoint archive: {keypoint_archive()}")
        print(f"  vocabulary size : {len(load_vocabulary())}")
    except Exception as exc:
        problems.append(f"dataset access: {exc}")
        print(f"  dataset access FAILED: {exc}")

    try:
        from backend.inference.audio import available_engines

        print(f"  TTS backends    : {available_engines()}")
    except Exception as exc:
        print(f"  TTS check failed: {exc}")

    cam_ok = False
    try:
        import cv2

        cap = cv2.VideoCapture(config.CAMERA_INDEX)
        cam_ok = bool(cap.isOpened())
        cap.release()
    except Exception:
        cam_ok = False
    print(f"  webcam available: {cam_ok} (expected False on servers/containers)")

    print("\n" + ("All checks passed." if not problems else "Problems:\n  - " + "\n  - ".join(problems)))
    return 0 if not problems else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command")

    def add(name, help_text):
        return sub.add_parser(name, help=help_text, add_help=False)

    add("info", "show configuration and artefacts")
    add("doctor", "environment self-check")
    add("camera", "webcam readiness check (devices, indices, display)")
    add("metadata", "fetch/verify official INCLUDE-50 metadata")
    add("audit", "run the dataset audit")
    p_pre = add("preprocess", "build sequence tensors")
    p_tr = add("train", "train one or all models")
    p_ev = add("evaluate", "evaluate the three models")
    add("robustness", "perturbation experiments")
    add("threshold", "confidence threshold sweep")
    p_rt = add("realtime", "webcam recognition")
    p_bt = add("batch", "offline / batch inference")
    add("api", "FastAPI service")
    p_te = add("test", "pytest suite")

    # Subcommands own their detailed arguments. Preserve unknown flags and forward
    # them to the selected script instead of making documented commands fail here.
    args, extra = ap.parse_known_args(argv)
    command = args.command or "info"
    extra = list(extra)
    if extra and extra[0] == "--":
        extra.pop(0)

    if command == "info":
        return cmd_info(args)
    if command == "doctor":
        return cmd_doctor(args)
    if command == "camera":
        return _run("scripts/camera_check.py", extra)
    if command == "metadata":
        return _run("training/fetch_metadata.py", extra)
    if command == "audit":
        return _run("training/audit_dataset.py", extra)
    if command == "preprocess":
        return _run("training/preprocess.py", extra)
    if command == "train":
        models = config.AVAILABLE_MODELS
        for flag in ("--model",):
            if flag in extra:
                models = (extra[extra.index(flag) + 1],)
        if models == config.AVAILABLE_MODELS and "--model" not in extra:
            rc = 0
            for name in models:
                rc |= _run(f"training/train_{name}.py", extra)
            return rc
        name = models[0]
        rest = [a for i, a in enumerate(extra) if not (a == "--model" or (i and extra[i - 1] == "--model"))]
        return _run(f"training/train_{name}.py", rest)
    if command == "evaluate":
        return _run("training/evaluate_models.py", extra)
    if command == "robustness":
        return _run("training/robustness.py", extra)
    if command == "threshold":
        return _run("training/tune_threshold.py", extra)
    if command == "realtime":
        return _run("backend/inference/realtime.py", extra)
    if command == "batch":
        return _run("backend/inference/batch.py", extra)
    if command == "api":
        uvicorn_args = extra or ["--host", config.API_HOST, "--port", str(config.API_PORT)]
        return _run_module("uvicorn", ["backend.main:app", *uvicorn_args])
    if command == "test":
        return _run_module("pytest", ["tests", "-v", *extra])

    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
