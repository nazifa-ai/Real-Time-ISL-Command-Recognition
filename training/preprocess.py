#!/usr/bin/env python3
"""
PHASE 4 -- reusable preprocessing pipeline.

    VIDEO / KEYPOINTS
        -> landmark packing        (75 landmarks x 3 coords)
        -> geometry units fix      (measured; see backend/preprocessing/geometry.py)
        -> per-frame normalisation (origin = mid-shoulder, scale = shoulder width)
        -> fixed-length sequence   (SEQUENCE_LENGTH frames, linear resample)
        -> optional landmark mask  (config.MASK_ENABLED)
        -> .npz tensors + class mapping

This module is the *only* place that turns dataset rows into model input, so the
three models provably receive identical data.  It is also used by
``training/robustness.py`` and by the live pipeline (through
``backend.preprocessing.landmarks`` / ``backend.preprocessing.sequence``), which is why the
geometry and normalisation code lives in ``backend/preprocessing/`` and not here.

Outputs (datasets/processed/)
----------------------------
    sequences_<protocol>.npz     X_train/X_val/X_test + y_* + clip metadata
    preprocess_summary.json      sizes, timings, skipped clips, label mapping hash

Usage
-----
    python training/preprocess.py                       # primary protocol
    python training/preprocess.py --protocol official   # comparability split
    python training/preprocess.py --augment-copies 2    # extra train copies
    python training/preprocess.py --limit 50            # smoke test
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from backend.preprocessing import include50, landmarks as lm, sequence as seq_ops  # noqa: E402

LOG = logging.getLogger("preprocess")


# --------------------------------------------------------------------------
# Core
# --------------------------------------------------------------------------
def clip_to_sequence(coords: np.ndarray, mask: np.ndarray,
                     length: Optional[int] = None) -> Optional[np.ndarray]:
    """(T,75,3) + (T,75) -> (length, 225) normalised feature sequence.

    Returns ``None`` when the clip carries fewer than ``config.MIN_VALID_FRAMES``
    frames that contain any landmark at all (such clips are unusable and are
    counted in the summary rather than silently zero-padded).
    """
    length = int(length or config.SEQUENCE_LENGTH)
    if coords.ndim != 3 or coords.shape[0] == 0:
        return None
    usable = (np.asarray(mask) > 0).any(axis=1)
    if int(usable.sum()) < config.MIN_VALID_FRAMES:
        return None

    feats, _ = lm.normalise_sequence(coords.astype(np.float32), np.asarray(mask, dtype=np.float32))
    feats = seq_ops.resample_sequence(feats, length, method=config.SEQ_RESAMPLE_METHOD)
    if feats.shape != (length, config.MODEL_INPUT_DIM if config.MASK_ENABLED else config.FEATURE_DIM):
        return None
    return feats.astype(np.float32)


def build_arrays(index, split: str, augment_copies: int = 0, seed: int = config.RANDOM_SEED,
                 limit: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, List[Dict], List[Dict]]:
    """Build (X, y) arrays for one split plus per-clip metadata.

    Augmentation is applied *offline* to the training split only, with a seeded
    RNG, so that every model sees byte-identical inputs.  Validation and test are
    never augmented.
    """
    rows = index[index["split"] == split].to_dict("records")
    if limit:
        rows = rows[:limit]
    classes = {c: i for i, c in enumerate(include50.load_vocabulary())}

    X: List[np.ndarray] = []
    y: List[int] = []
    meta: List[Dict] = []
    skipped: List[Dict] = []

    rng = np.random.default_rng(seed + {"train": 0, "val": 1, "test": 2}.get(split, 3))
    for row in rows:
        try:
            coords, mask, info = include50.read_keypoints_for(row["FilePath"])
        except Exception as exc:
            skipped.append({"filepath": row["FilePath"], "reason": f"{type(exc).__name__}: {exc}"})
            continue
        seq = clip_to_sequence(coords, mask)
        if seq is None:
            skipped.append({"filepath": row["FilePath"], "reason": "too few usable frames"})
            continue
        label = classes[row["label"]]
        copies = [seq] + [seq_ops.augment(seq, rng) for _ in range(augment_copies)] if split == "train" else [seq]
        for k, s in enumerate(copies):
            X.append(s)
            y.append(label)
            meta.append({
                "filepath": row["FilePath"], "label": row["label"], "class_index": label,
                "split": split, "copy": k, "frames_source": int(coords.shape[0]),
                "x_axis_scale": float(info.get("x_axis_scale", 1.0)),
            })
    if not X:
        return (np.zeros((0, config.SEQUENCE_LENGTH, config.FEATURE_DIM), dtype=np.float32),
                np.zeros((0,), dtype=np.int64), meta, skipped)
    return np.stack(X).astype(np.float32), np.array(y, dtype=np.int64), meta, skipped


def label_mapping() -> Dict[str, object]:
    vocab = include50.load_vocabulary()
    payload = {"classes": vocab, "index": {c: i for i, c in enumerate(vocab)},
               "num_classes": len(vocab)}
    payload["sha1"] = hashlib.sha1("|".join(vocab).encode()).hexdigest()
    return payload


def run(protocol: str, augment_copies: int, seed: int, limit: Optional[int],
        out_dir: Optional[Path] = None) -> Path:
    out_dir = Path(out_dir or config.PROCESSED_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    index, split_summary = include50.build_index(protocol=protocol)
    mapping = label_mapping()
    LOG.info("protocol=%s  train=%d val=%d test=%d  classes=%d",
             protocol, split_summary.train, split_summary.val, split_summary.test, mapping["num_classes"])

    t0 = time.time()
    arrays: Dict[str, np.ndarray] = {}
    all_meta: Dict[str, List[Dict]] = {}
    all_skipped: Dict[str, List[Dict]] = {}
    for split, copies in (("train", augment_copies), ("val", 0), ("test", 0)):
        X, y, meta, skipped = build_arrays(index, split, augment_copies=copies, seed=seed, limit=limit)
        arrays[f"X_{split}"] = X
        arrays[f"y_{split}"] = y
        all_meta[split] = meta
        all_skipped[split] = skipped
        LOG.info("  %-5s X=%s y=%s  (skipped %d)", split, X.shape, y.shape, len(skipped))

    npz_path = out_dir / f"sequences_{protocol}.npz"
    np.savez_compressed(
        npz_path,
        **arrays,
        classes=np.array(mapping["classes"], dtype=object),
        meta_json=np.array(json.dumps(all_meta, default=str)),
        skipped_json=np.array(json.dumps(all_skipped, default=str)),
    )

    summary = {
        "protocol": protocol,
        "sequence_length": config.SEQUENCE_LENGTH,
        "feature_dim": config.FEATURE_DIM,
        "model_input_dim": config.MODEL_INPUT_DIM,
        "mask_enabled": config.MASK_ENABLED,
        "augment_copies": augment_copies,
        "seed": seed,
        "label_sha1": mapping["sha1"],
        "num_classes": mapping["num_classes"],
        "shapes": {k: list(v.shape) for k, v in arrays.items()},
        "skipped": {k: len(v) for k, v in all_skipped.items()},
        "skipped_detail": all_skipped,
        "seconds": round(time.time() - t0, 2),
        "normalisation": {
            "origin": "mid-shoulder (fallback: mid-hip, then landmark centroid)",
            "scale": "shoulder width (fallback: hip width, then 1.0)",
            "x_axis_scale": config.X_AXIS_SCALE,
            "axis_correction_applied": config.APPLY_KEYPOINT_AXIS_CORRECTION,
        },
        "sequence": {"resample": config.SEQ_RESAMPLE_METHOD, "pad_mode": config.SEQ_PAD_MODE,
                     "min_valid_frames": config.MIN_VALID_FRAMES},
        "augmentation": dict(config.AUGMENTATION),
        "split_summary": split_summary.as_dict(),
    }
    (out_dir / f"preprocess_summary_{protocol}.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "label_mapping.json").write_text(json.dumps(mapping, indent=2))
    LOG.info("wrote %s (%.1f MB)", npz_path, npz_path.stat().st_size / 1e6)
    return npz_path


def load_processed(protocol: Optional[str] = None, path: Optional[Path] = None) -> Dict[str, np.ndarray]:
    protocol = protocol or config.SPLIT_PROTOCOL
    path = Path(path or (Path(config.PROCESSED_DIR) / f"sequences_{protocol}.npz"))
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run: python training/preprocess.py --protocol {protocol}")
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--protocol", default=config.SPLIT_PROTOCOL, choices=["session_disjoint", "official", "hf"])
    ap.add_argument("--augment-copies", type=int, default=_def_copies() if False else None,
                    help="extra augmented copies of each training clip (default from config)")
    ap.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    ap.add_argument("--limit", type=int, default=None, help="limit clips per split (smoke test)")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    copies = args.augment_copies
    if copies is None:
        copies = int(config.AUGMENTATION.get("augment_copies", 1)) if config.AUGMENTATION.get("enabled", True) else 0
    try:
        run(args.protocol, copies, args.seed, args.limit, Path(args.out_dir) if args.out_dir else None)
    except Exception:
        traceback.print_exc()
        return 1
    return 0


def _def_copies():  # pragma: no cover - helper for argparse default
    return None


if __name__ == "__main__":
    raise SystemExit(main())
