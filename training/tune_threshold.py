#!/usr/bin/env python3
"""
Confidence-threshold study (why 0.70 is a starting point, not an optimum).

The gate decides which predictions are shown/captioned and which are reported as
UNCERTAIN.  Sweeping it changes the trade-off between *coverage* (how often the
system commits to a sign) and *selective accuracy* (how often a committed sign is
right).  This script measures that trade-off on the held-out split for every model
and writes:

    results/threshold_sweep.csv    one row per (protocol, model, threshold)
    results/threshold_sweep.png    coverage vs selective accuracy curves

Nothing here is used to silently retune the shipped default; it exists so the
choice is documented and reproducible, and so a user can pick a gate that matches
their own tolerance (see docs/evaluation.md).

Usage
-----
    python training/tune_threshold.py
    python training/tune_threshold.py --split val --models gru_mha
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from training import preprocess as prep  # noqa: E402
from training.evaluate_models import load_model  # noqa: E402

LOG = logging.getLogger("threshold")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

THRESHOLDS = [round(t, 2) for t in np.arange(0.20, 0.96, 0.05)]


def sweep(protocol: str, model_names: Sequence[str], split: str = "val") -> List[Dict[str, object]]:
    data = prep.load_processed(protocol)
    X, y = data[f"X_{split}"], data[f"y_{split}"]
    rows: List[Dict[str, object]] = []
    for name in model_names:
        model, path = load_model(name)
        probs = model.predict(X.astype(np.float32), batch_size=config.BATCH_SIZE, verbose=0)
        conf = probs.max(axis=1)
        pred = probs.argmax(axis=1)
        correct = (pred == y).astype(np.float64)

        for t in THRESHOLDS:
            mask = conf >= t
            coverage = float(mask.mean())
            selective = float(correct[mask].mean()) if mask.any() else None
            rows.append({
                "protocol": protocol, "split": split, "model": name, "threshold": t,
                "coverage": round(coverage, 6),
                "selective_accuracy": (round(selective, 6) if selective is not None else None),
                "uncertain_rate": round(1.0 - coverage, 6),
                "accepted": int(mask.sum()),
                "accepted_correct": int(correct[mask].sum()),
                # accuracy if every rejected sample were counted wrong (a pessimistic bound)
                "pessimistic_accuracy": round(float(correct[mask].sum()) / max(1, len(y)), 6),
            })
        LOG.info("%s: sweep done (%d thresholds)", name, len(THRESHOLDS))
    return rows


def plot_sweep(rows: List[Dict[str, object]], out: Path, suffix: str = "") -> Path:
    models = sorted({str(r["model"]) for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for name in models:
        sub = [r for r in rows if r["model"] == name]
        t = [float(r["threshold"]) for r in sub]
        axes[0].plot(t, [float(r["coverage"]) for r in sub], marker="o", ms=3, label=name)
        sel = [(float(r["selective_accuracy"]) if r["selective_accuracy"] is not None else np.nan)
               for r in sub]
        axes[1].plot(t, sel, marker="o", ms=3, label=name)
    axes[0].axvline(config.CONFIDENCE_THRESHOLD, ls="--", c="grey")
    axes[1].axvline(config.CONFIDENCE_THRESHOLD, ls="--", c="grey",
                    label=f"shipped default {config.CONFIDENCE_THRESHOLD:.2f}")
    axes[0].set_title("Coverage vs confidence gate")
    axes[0].set_ylabel("fraction of clips above the gate")
    axes[1].set_title("Selective accuracy vs confidence gate")
    axes[1].set_ylabel("accuracy among accepted clips")
    for ax in axes:
        ax.set_xlabel("confidence threshold")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def run(results_dir: Path, protocols: Sequence[str], model_names: Sequence[str],
        split: str = "val") -> Dict[str, object]:
    all_rows: List[Dict[str, object]] = []
    for protocol in protocols:
        all_rows += sweep(protocol, model_names, split=split)

    csv_path = results_dir / "threshold_sweep.csv"
    fields = ["protocol", "split", "model", "threshold", "coverage", "selective_accuracy",
              "uncertain_rate", "accepted", "accepted_correct", "pessimistic_accuracy"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        wr.writerows(all_rows)

    png = plot_sweep([r for r in all_rows if r["protocol"] == protocols[0]],
                     results_dir / "threshold_sweep.png")

    # what the shipped default actually buys, measured - not assumed
    at_default = [r for r in all_rows if abs(float(r["threshold"]) - config.CONFIDENCE_THRESHOLD) < 1e-9]
    summary = {
        "split": split,
        "shipped_default": config.CONFIDENCE_THRESHOLD,
        "thresholds_swept": THRESHOLDS,
        "at_default": at_default,
        "note": ("0.70 was chosen as a conservative starting point before any tuning; "
                 "this sweep is the evidence for/against it. It is configurable via "
                 "ISL_CONFIDENCE_THRESHOLD and per-profile within a bounded band."),
        "csv": str(csv_path), "plot": str(png),
    }
    (results_dir / "threshold_sweep_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return {"rows": all_rows, "summary": summary, "csv": csv_path, "png": png}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--protocols", nargs="+", default=[config.SPLIT_PROTOCOL, "official"])
    ap.add_argument("--models", nargs="+", default=list(config.AVAILABLE_MODELS))
    ap.add_argument("--split", default="val", choices=("val", "test"))
    ap.add_argument("--results-dir", default=str(config.RESULTS_DIR))
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    res = run(Path(args.results_dir), args.protocols, args.models, split=args.split)

    rows = [r for r in res["rows"] if r["protocol"] == args.protocols[0]]
    print(f"\n=== coverage / selective accuracy on the {args.split} split "
          f"({args.protocols[0]}) ===")
    print(f"{'model':<9}{'thr':>6}{'coverage':>10}{'sel.acc':>9}{'pess.acc':>10}")
    for r in rows:
        sel = f"{r['selective_accuracy']:.4f}" if r["selective_accuracy"] is not None else "  n/a"
        print(f"{r['model']:<9}{r['threshold']:>6.2f}{r['coverage']:>10.3f}{sel:>9}"
              f"{r['pessimistic_accuracy']:>10.4f}")
    print(f"\nwrote {res['csv']}\nwrote {res['png']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
