#!/usr/bin/env python3
"""
PHASE 8 -- robustness / degradation experiments.

Perturbations are applied to the *real* frozen test tensors and pushed through the
*real trained models*; nothing here is simulated or extrapolated.  Each condition
is evaluated with a fixed seed so a rerun reproduces the same numbers.

Conditions
----------
clean                     unmodified test set (baseline for every delta)
landmark_dropout_p10/p30  random landmarks set to "missing" (zeros) per frame
hand_left_missing         left-hand landmarks removed from every frame
hand_right_missing        right-hand landmarks removed from every frame
both_hands_missing        both hands removed (pose only: a deliberately harsh case)
gaussian_sigma_10/25/50   additive Gaussian noise on normalised coordinates
frame_drop_p20/p40/p60    random frames removed, sequence resampled back to 30
frame_skip_keep50         uniform temporal subsampling (1 frame in 2)
mask_missing              coordinates kept but presence mask zeroed (MASK_ENABLED builds only)

Outputs
-------
results/robustness_results.csv     condition, model, accuracy, macro_f1, weighted_f1, deltas, params
results/robustness_degradation.png accuracy and macro-F1 per condition, per model

Usage
-----
    python training/robustness.py
    python training/robustness.py --protocol official --suffix _official
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import zlib
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from backend.preprocessing import sequence as seq_ops  # noqa: E402
from training import models as model_zoo  # noqa: E402
from training import preprocess as prep  # noqa: E402
from training.evaluate_models import confusion_matrix, load_model, per_class_metrics  # noqa: E402

LOG = logging.getLogger("robustness")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FEATURE_DIM = 225
NUM_LM = 75
LEFT_SLICE = slice(33 * 3, 54 * 3)      # 21 left-hand landmarks x 3 coords
RIGHT_SLICE = slice(54 * 3, 75 * 3)


# --------------------------------------------------------------------------
# perturbations:  (N, T, D) -> (N, T, D)
# --------------------------------------------------------------------------
def p_clean(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return X.copy()


def _dropout(X: np.ndarray, prob: float, rng: np.random.Generator) -> np.ndarray:
    out = X.copy()
    n, t, d = out.shape
    coords = out.reshape(n, t, NUM_LM, 3)
    keep = rng.random((n, t, NUM_LM)) >= prob
    coords[~keep] = 0.0
    return out


def p_landmark_dropout_p10(X, rng):
    return _dropout(X, 0.10, rng)


def p_landmark_dropout_p30(X, rng):
    return _dropout(X, 0.30, rng)


def _missing_hand(X: np.ndarray, sl: slice) -> np.ndarray:
    out = X.copy()
    out[:, :, sl] = 0.0
    return out


def p_hand_left_missing(X, rng):
    return _missing_hand(X, LEFT_SLICE)


def p_hand_right_missing(X, rng):
    return _missing_hand(X, RIGHT_SLICE)


def p_both_hands_missing(X, rng):
    return _missing_hand(X, slice(33 * 3, 75 * 3))


def _gaussian(X: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    noise = rng.normal(0.0, sigma, size=X.shape).astype(np.float32)
    return X + noise


def p_gaussian_sigma_10(X, rng):
    return _gaussian(X, 0.10, rng)          # 10% of one shoulder-width


def p_gaussian_sigma_25(X, rng):
    return _gaussian(X, 0.25, rng)


def p_gaussian_sigma_50(X, rng):
    return _gaussian(X, 0.50, rng)


def _frame_drop(X: np.ndarray, prob: float, rng: np.random.Generator) -> np.ndarray:
    """Remove frames at random (simulating detector lag) and resample to the original length."""
    n, t, d = X.shape
    out = np.zeros_like(X)
    for i in range(n):
        keep_idx = [f for f in range(t) if rng.random() >= prob]
        if len(keep_idx) < config.MIN_VALID_FRAMES:
            keep_idx = list(range(t))
        out[i] = seq_ops.resample_sequence(X[i][keep_idx], t, method=config.SEQ_RESAMPLE_METHOD)
    return out


def p_frame_drop_p20(X, rng):
    return _frame_drop(X, 0.20, rng)


def p_frame_drop_p40(X, rng):
    return _frame_drop(X, 0.40, rng)


def p_frame_drop_p60(X, rng):
    return _frame_drop(X, 0.60, rng)


def p_frame_skip_keep50(X, rng):
    n, t, d = X.shape
    out = np.zeros_like(X)
    for i in range(n):
        sub = seq_ops.frame_skip(X[i], keep_ratio=0.5)
        out[i] = seq_ops.resample_sequence(sub, t, method=config.SEQ_RESAMPLE_METHOD)
    return out


def p_mask_missing(X, rng):
    """Zero the presence mask (frames 225:300) if the build uses it; otherwise a no-op copy."""
    out = X.copy()
    if config.MASK_ENABLED and out.shape[-1] > FEATURE_DIM:
        out[:, :, FEATURE_DIM:] = 0.0
    return out


CONDITIONS: Sequence[Tuple[str, str, Callable]] = (
    ("clean", "no perturbation", p_clean),
    ("landmark_dropout_p10", "10% of landmarks dropped per frame", p_landmark_dropout_p10),
    ("landmark_dropout_p30", "30% of landmarks dropped per frame", p_landmark_dropout_p30),
    ("hand_left_missing", "left-hand landmarks removed", p_hand_left_missing),
    ("hand_right_missing", "right-hand landmarks removed", p_hand_right_missing),
    ("both_hands_missing", "both hands removed (pose only)", p_both_hands_missing),
    ("gaussian_sigma_10", "Gaussian jitter sigma=0.10 shoulder-widths", p_gaussian_sigma_10),
    ("gaussian_sigma_25", "Gaussian jitter sigma=0.25 shoulder-widths", p_gaussian_sigma_25),
    ("gaussian_sigma_50", "Gaussian jitter sigma=0.50 shoulder-widths", p_gaussian_sigma_50),
    ("frame_drop_p20", "20% of frames dropped, resampled", p_frame_drop_p20),
    ("frame_drop_p40", "40% of frames dropped, resampled", p_frame_drop_p40),
    ("frame_drop_p60", "60% of frames dropped, resampled", p_frame_drop_p60),
    ("frame_skip_keep50", "uniform subsampling, every 2nd frame", p_frame_skip_keep50),
    ("mask_missing", "presence mask zeroed (MASK_ENABLED builds)", p_mask_missing),
)


# --------------------------------------------------------------------------
def evaluate_condition(model, X: np.ndarray, y: np.ndarray, k: int) -> Dict[str, float]:
    probs = model.predict(X.astype(np.float32), batch_size=config.BATCH_SIZE, verbose=0)
    pred = np.argmax(probs, axis=1)
    cm = confusion_matrix(y, pred, k)
    m = per_class_metrics(cm)
    support = m["support"].astype(np.float64)
    weights = support / max(1.0, support.sum())
    present = support > 0
    return {
        "accuracy": float(np.trace(cm) / max(1, cm.sum())),
        "macro_precision": float(m["precision"][present].mean()) if present.any() else 0.0,
        "macro_recall": float(m["recall"][present].mean()) if present.any() else 0.0,
        "macro_f1": float(m["f1"][present].mean()) if present.any() else 0.0,
        "weighted_f1": float((m["f1"] * weights).sum()),
        "mean_confidence": float(np.mean(np.max(probs, axis=1))),
        "n_below_threshold": int(np.sum(np.max(probs, axis=1) < config.CONFIDENCE_THRESHOLD)),
    }


def plot_degradation(rows: List[Dict[str, object]], out: Path, suffix: str = "") -> Path:
    models = sorted({str(r["model"]) for r in rows})
    conds = [c for c, _, _ in CONDITIONS if c != "clean"]
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=False)
    width = 0.8 / max(1, len(models))
    x = np.arange(len(conds))
    for i, m in enumerate(models):
        acc = [float(next(r["accuracy"] for r in rows if r["model"] == m and r["condition"] == c)) for c in conds]
        f1 = [float(next(r["macro_f1"] for r in rows if r["model"] == m and r["condition"] == c)) for c in conds]
        axes[0].bar(x + i * width, acc, width, label=m)
        axes[1].bar(x + i * width, f1, width, label=m)
    for ax, title in ((axes[0], "Top-1 accuracy under perturbation"),
                      (axes[1], "Macro-F1 under perturbation")):
        ax.set_xticks(x + 0.4 - width / 2)
        ax.set_xticklabels(conds, rotation=35, ha="right", fontsize=8)
        ax.set_ylim(0, 1.0)
        ax.set_title(title + suffix)
        ax.grid(axis="y", alpha=0.3)
        ax.legend()
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def run(protocol: str, model_names: Sequence[str], results_dir: Path, suffix: str = "",
        seed: int = config.RANDOM_SEED, split: str = "test") -> Dict[str, object]:
    data = prep.load_processed(protocol)
    X = data[f"X_{split}"]
    y = data[f"y_{split}"]
    classes = [str(c) for c in data["classes"]]
    k = len(classes)

    rows: List[Dict[str, object]] = []
    started = time.time()
    for name in model_names:
        model, path = load_model(name)
        params = model_zoo.parameter_count(model)
        LOG.info("=== %s (%s) ===", name, path.name)
        baseline: Optional[Dict[str, float]] = None
        for condition, description, fn in CONDITIONS:
            rng = np.random.default_rng(seed + zlib.crc32(f"{condition}|{name}|{protocol}".encode()) % 100000)
            Xp = fn(X, rng)
            if Xp.shape != X.shape:
                raise ValueError(f"{condition}: perturbation changed shape {Xp.shape} != {X.shape}")
            metrics = evaluate_condition(model, Xp, y, k)
            if condition == "clean":
                baseline = metrics
            assert baseline is not None
            row = {
                "protocol": protocol, "split": split, "model": name, "condition": condition,
                "description": description, "parameters": params,
                **{key: round(float(val), 6) for key, val in metrics.items()},
                "accuracy_delta": round(metrics["accuracy"] - baseline["accuracy"], 6),
                "macro_f1_delta": round(metrics["macro_f1"] - baseline["macro_f1"], 6),
                "relative_accuracy_drop_pct": round(
                    100.0 * (baseline["accuracy"] - metrics["accuracy"]) / max(1e-9, baseline["accuracy"]), 3),
            }
            rows.append(row)
            LOG.info("  %-22s acc=%.4f (d=%+.4f)  macro_f1=%.4f (d=%+.4f)  unc<gate=%.1f%%",
                     condition, row["accuracy"], row["accuracy_delta"], row["macro_f1"],
                     row["macro_f1_delta"], 100.0 * row["n_below_threshold"] / max(1, len(y)))
            print(f"{name:<8} {condition:<22} acc={row['accuracy']:.4f} "
                  f"delta={row['accuracy_delta']:+.4f} macro_f1={row['macro_f1']:.4f} "
                  f"below_gate={100.0 * row['n_below_threshold'] / max(1, len(y)):.1f}%")

    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / f"robustness_results{suffix}.csv"
    fields = ["protocol", "split", "model", "condition", "description", "parameters",
              "accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1",
              "mean_confidence", "n_below_threshold", "accuracy_delta", "macro_f1_delta",
              "relative_accuracy_drop_pct"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        wr.writerows(rows)

    plot_path = plot_degradation(rows, results_dir / f"robustness_degradation{suffix}.png", suffix)

    summary = {
        "protocol": protocol, "split": split, "seed": seed, "n_clips": int(len(y)),
        "conditions": [{"condition": c, "description": d} for c, d, _ in CONDITIONS],
        "best_model_per_condition": {
            c: max((r for r in rows if r["condition"] == c), key=lambda r: float(r["accuracy"]))["model"]  # type: ignore[arg-type]
            for c, _, _ in CONDITIONS
        },
        "seconds": round(time.time() - started, 1),
        "csv": str(csv_path), "plot": str(plot_path),
    }
    (results_dir / f"robustness_summary{suffix}.json").write_text(json.dumps(summary, indent=2))
    return {"rows": rows, "summary": summary, "csv": csv_path, "plot": plot_path}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--protocol", default=config.SPLIT_PROTOCOL,
                    choices=("session_disjoint", "official", "hf"))
    ap.add_argument("--models", nargs="+", default=list(config.AVAILABLE_MODELS))
    ap.add_argument("--results-dir", default=str(config.RESULTS_DIR))
    ap.add_argument("--suffix", default=None)
    ap.add_argument("--split", default="test", choices=("val", "test"))
    ap.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    suffix = args.suffix if args.suffix is not None else (
        "" if args.protocol == config.SPLIT_PROTOCOL else f"_{args.protocol}")
    res = run(args.protocol, args.models, Path(args.results_dir), suffix=suffix,
              seed=args.seed, split=args.split)
    print(f"\nwrote {res['csv']}\nwrote {res['plot']}\nwrote " +
          str(Path(args.results_dir) / f"robustness_summary{suffix}.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
