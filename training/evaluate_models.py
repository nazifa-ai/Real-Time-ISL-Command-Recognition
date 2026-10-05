#!/usr/bin/env python3
"""
PHASE 7 -- evaluation and three-model comparison.

For every trained model (LSTM, GRU, GRU+MHA) this script evaluates the *same*
frozen test set and writes:

    results/per_class_comparison.csv      class, <model>_precision/recall/f1/support
    results/summary_metrics.csv           accuracy, macro/weighted P/R/F1, params, latency, FPS
    results/confusion_<model>.png         raw counts + row-normalised confusion matrices
    results/model_comparison.png          accuracy / macro-F1 / weighted-F1 bars
    results/latency_comparison.png        single-sample latency and throughput
    results/evaluation_summary_<protocol>.json   full detail (top confusions, failures)

Fairness: no augmentation or test-time tricks are applied here.  Every model sees
exactly ``X_test`` from the same ``sequences_<protocol>.npz`` and the same class
mapping, so the only difference between columns is the architecture.

Usage
-----
    python training/evaluate_models.py                      # default protocol
    python training/evaluate_models.py --protocol official  # leaked-split comparison
    python training/evaluate_models.py --models lstm gru_mha
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from training import models as model_zoo  # noqa: E402
from training import preprocess as prep  # noqa: E402

LOG = logging.getLogger("evaluate")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# --------------------------------------------------------------------------
# metrics (computed directly - no dependency on sklearn for the core numbers)
# --------------------------------------------------------------------------
def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> np.ndarray:
    cm = np.zeros((k, k), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def per_class_metrics(cm: np.ndarray) -> Dict[str, np.ndarray]:
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(tp), where=(precision + recall) > 0)
    support = cm.sum(axis=1).astype(np.int64)
    return {"precision": precision, "recall": recall, "f1": f1, "support": support}


def aggregate(m: Dict[str, np.ndarray], y_true: np.ndarray) -> Dict[str, float]:
    n = int(y_true.size)
    present = m["support"] > 0
    macro = {k: float(m[k][present].mean()) if present.any() else 0.0
             for k in ("precision", "recall", "f1")}
    weights = m["support"].astype(np.float64) / max(1, n)
    weighted = {k: float((m[k] * weights).sum()) for k in ("precision", "recall", "f1")}
    return {
        "macro_precision": macro["precision"], "macro_recall": macro["recall"], "macro_f1": macro["f1"],
        "weighted_precision": weighted["precision"], "weighted_recall": weighted["recall"],
        "weighted_f1": weighted["f1"], "n_samples": n,
    }


# --------------------------------------------------------------------------
# plotting
# --------------------------------------------------------------------------
def plot_confusion(cm: np.ndarray, classes: Sequence[str], model_name: str, out: Path) -> Path:
    acc = float(np.trace(cm) / max(1, cm.sum()))
    norm = cm.astype(np.float64) / np.maximum(1, cm.sum(axis=1, keepdims=True))
    fig, axes = plt.subplots(1, 2, figsize=(22, 10))
    for ax, mat, title, fmt in ((axes[0], cm, f"{model_name}: counts (acc={acc:.3f})", "d"),
                                (axes[1], norm, f"{model_name}: row-normalised", ".2f")):
        im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=(mat.max() if mat.max() > 0 else 1))
        ax.set_xticks(range(len(classes)))
        ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=90, fontsize=5)
        ax.set_yticklabels(classes, fontsize=5)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(title, fontsize=11)
        fig.colorbar(im, ax=ax, fraction=0.046)
        lim = len(classes)
        if lim <= 50:
            for i in range(lim):
                for j in range(lim):
                    v = mat[i, j]
                    if v:
                        txt = f"{int(v)}" if fmt == "d" else f"{v:.2f}"
                        ax.text(j, i, txt, ha="center", va="center", fontsize=3.6,
                                color="white" if mat[i, j] > mat.max() * 0.6 else "black")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_model_comparison(summary: Dict[str, Dict[str, float]], out: Path) -> Path:
    names = list(summary.keys())
    metrics = ["accuracy", "macro_precision", "macro_recall", "macro_f1",
               "weighted_precision", "weighted_recall", "weighted_f1"]
    x = np.arange(len(metrics))
    w = 0.8 / max(1, len(names))
    fig, ax = plt.subplots(figsize=(12, 5.5))
    for i, n in enumerate(names):
        vals = [summary[n].get(m, 0.0) for m in metrics]
        bars = ax.bar(x + i * w, vals, w, label=n)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.3f}",
                    ha="center", fontsize=6.5, rotation=90)
    ax.set_xticks(x + 0.4 - w / 2)
    ax.set_xticklabels([m.replace("_", "\n") for m in metrics], fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score")
    ax.set_title("LSTM vs GRU vs GRU+MHA on the identical frozen test set")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_latency(summary: Dict[str, Dict[str, float]], out: Path) -> Path:
    names = list(summary.keys())
    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    lat = [summary[n].get("latency_ms_mean", 0.0) for n in names]
    p95 = [summary[n].get("latency_ms_p95", 0.0) for n in names]
    fps = [summary[n].get("throughput_fps", 0.0) for n in names]
    axes[0].bar(x - 0.2, lat, 0.4, label="mean latency (ms)")
    axes[0].bar(x + 0.2, p95, 0.4, label="p95 latency (ms)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names)
    axes[0].set_ylabel("milliseconds")
    axes[0].set_title("Single-clip inference latency (CPU)")
    axes[0].legend()
    for i, v in enumerate(lat):
        axes[0].text(i - 0.2, v, f"{v:.1f}", ha="center", va="bottom", fontsize=7)
    for i, v in enumerate(p95):
        axes[0].text(i + 0.2, v, f"{v:.1f}", ha="center", va="bottom", fontsize=7)
    axes[1].bar(x, fps, 0.5, color="tab:green")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names)
    axes[1].set_ylabel("sequences / second")
    axes[1].set_title("Throughput")
    for i, v in enumerate(fps):
        axes[1].text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=7)
    for ax in axes:
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_history(model_name: str, out: Path) -> Optional[Path]:
    """Training-history curves straight from ``models/<name>/training_history.json``."""
    p = Path(config.MODEL_PATHS[model_name]) / "training_history.json"
    if not p.exists():
        return None
    h = json.loads(p.read_text())["history"]
    epochs = range(1, len(h.get("loss", [])) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    axes[0].plot(epochs, h.get("accuracy", []), label="train")
    axes[0].plot(epochs, h.get("val_accuracy", []), label="val")
    axes[0].set_title(f"{model_name}: accuracy")
    axes[0].set_xlabel("epoch")
    axes[0].legend()
    axes[1].plot(epochs, h.get("loss", []), label="train")
    axes[1].plot(epochs, h.get("val_loss", []), label="val")
    axes[1].set_title(f"{model_name}: loss")
    axes[1].set_xlabel("epoch")
    axes[1].legend()
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------
# model loading / timing
# --------------------------------------------------------------------------
def load_model(model_name: str):
    import tensorflow as tf

    for fname in ("model.keras", "best.keras"):
        p = Path(config.MODEL_PATHS[model_name]) / fname
        if p.exists():
            return tf.keras.models.load_model(str(p)), p
    raise FileNotFoundError(
        f"no trained model in {config.MODEL_PATHS[model_name]} - run: python training/train_{model_name}.py"
    )


def measure_latency(model, X: np.ndarray, repeats: int = 3) -> Dict[str, float]:
    """Single-sample latency (what real-time inference pays) + batched throughput."""
    sample = X[:1].astype(np.float32)
    model.predict_on_batch(sample)                     # warm-up, excluded from timing
    per_sample: List[float] = []
    for i in range(min(repeats, len(X))):
        t0 = time.perf_counter()
        model.predict_on_batch(X[i:i + 1].astype(np.float32))
        per_sample.append((time.perf_counter() - t0) * 1000.0)

    t0 = time.perf_counter()
    model.predict_on_batch(X.astype(np.float32))
    batch_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "latency_ms_mean": float(np.mean(per_sample)),
        "latency_ms_p50": float(np.percentile(per_sample, 50)),
        "latency_ms_p95": float(np.percentile(per_sample, 95)),
        "latency_ms_batch_per_sample": batch_ms / max(1, len(X)),
        "throughput_fps": float(1000.0 / np.mean(per_sample)) if per_sample else 0.0,
        "batch_throughput_fps": float(len(X) / (batch_ms / 1000.0)) if batch_ms > 0 else 0.0,
    }


def measure_per_class_latency(model, X: np.ndarray, y: np.ndarray,
                              classes: Sequence[str]) -> Dict[str, Dict[str, float]]:
    """Measure one warmed, real single-sequence inference per held-out test clip.

    Per-class means are descriptive timing of the actual examples in that class,
    not a claim that the output label changes model compute cost. Small support is
    preserved for the UI to show beside every number.
    """
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    model.predict_on_batch(X[:1])
    elapsed_ms = np.empty(len(X), dtype=np.float64)
    for i in range(len(X)):
        start = time.perf_counter()
        model.predict_on_batch(X[i:i + 1])
        elapsed_ms[i] = (time.perf_counter() - start) * 1000.0
    result: Dict[str, Dict[str, float]] = {}
    for i, label in enumerate(classes):
        values = elapsed_ms[y == i]
        result[str(label)] = {
            "mean_ms": float(values.mean()) if values.size else 0.0,
            "p50_ms": float(np.percentile(values, 50)) if values.size else 0.0,
            "samples": int(values.size),
        }
    return result


# --------------------------------------------------------------------------
# main evaluation
# --------------------------------------------------------------------------
def evaluate(protocol: str, model_names: Sequence[str], out_dir: Path,
             results_dir: Path, suffix: str = "", repeats: int = 3,
             split: str = "test") -> Dict[str, object]:
    data = prep.load_processed(protocol)
    X = data.get(f"X_{split}")
    y = data.get(f"y_{split}")
    classes = [str(c) for c in data["classes"]]
    k = len(classes)
    if X is None or y is None:
        raise KeyError(f"npz has no X_{split}/y_{split}")

    LOG.info("protocol=%s split=%s X=%s classes=%d", protocol, split, X.shape, k)

    per_class: Dict[str, Dict[str, np.ndarray]] = {}
    summary_rows: Dict[str, Dict[str, object]] = {}
    detail: Dict[str, object] = {"protocol": protocol, "split": split,
                                 "n_samples": int(len(y)), "classes": classes,
                                 "models": {}}

    for name in model_names:
        model, path = load_model(name)
        params = model_zoo.parameter_count(model)
        probs = model.predict(X.astype(np.float32), batch_size=config.BATCH_SIZE, verbose=0)
        pred = np.argmax(probs, axis=1)
        cm = confusion_matrix(y, pred, k)
        m = per_class_metrics(cm)
        agg = aggregate(m, y)
        acc = float(np.trace(cm) / max(1, cm.sum()))
        timing = measure_latency(model, X, repeats=repeats)
        class_timing = measure_per_class_latency(model, X, y, classes)

        per_class[name] = m
        summary_rows[name] = {
            "model": name, "protocol": protocol, "split": split, "n_samples": int(len(y)),
            "n_classes": k, "parameters": params, "accuracy": acc,
            "macro_precision": agg["macro_precision"], "macro_recall": agg["macro_recall"],
            "macro_f1": agg["macro_f1"], "weighted_precision": agg["weighted_precision"],
            "weighted_recall": agg["weighted_recall"], "weighted_f1": agg["weighted_f1"],
            "latency_ms_mean": timing["latency_ms_mean"], "latency_ms_p50": timing["latency_ms_p50"],
            "latency_ms_p95": timing["latency_ms_p95"],
            "latency_ms_batch_per_sample": timing["latency_ms_batch_per_sample"],
            "throughput_fps": timing["throughput_fps"],
            "batch_throughput_fps": timing["batch_throughput_fps"],
        }

        plot_confusion(cm, classes, name, results_dir / f"confusion_{name}{suffix}.png")
        plot_history(name, results_dir / f"training_history_{name}{suffix}.png")

        # --- error analysis: most frequent confusions + worst classes -----------
        off = [(classes[i], classes[j], int(cm[i, j]))
               for i in range(k) for j in range(k) if i != j and cm[i, j] > 0]
        off.sort(key=lambda t: -t[2])
        weakest = sorted(
            [{"class": classes[i], "recall": float(m["recall"][i]),
              "precision": float(m["precision"][i]), "f1": float(m["f1"][i]),
              "support": int(m["support"][i])}
             for i in range(k) if m["support"][i] > 0],
            key=lambda d: d["f1"],
        )
        detail["models"][name] = {          # type: ignore[index]
            "weights": str(path),
            "metrics": summary_rows[name],
            "per_class_latency": class_timing,
            "top_confusions": [{"true": t, "predicted": p, "count": c} for t, p, c in off[:20]],
            "weakest_classes": weakest[:10],
            "strongest_classes": weakest[-5:][::-1],
            "confusion_matrix": cm.tolist(),
        }
        LOG.info("%-8s acc=%.4f macro_f1=%.4f weighted_f1=%.4f params=%d lat=%.1fms",
                 name, acc, agg["macro_f1"], agg["weighted_f1"], params, timing["latency_ms_mean"])

    # ---------------- per_class_comparison.csv (wide format) ----------------
    header = ["class"]
    for name in model_names:
        header += [f"{name}_precision", f"{name}_recall", f"{name}_accuracy", f"{name}_f1", f"{name}_support",
                   f"{name}_latency_ms_mean", f"{name}_latency_samples"]
    rows = []
    for i, cname in enumerate(classes):
        row = {"class": cname}
        for name in model_names:
            m = per_class[name]
            row[f"{name}_precision"] = round(float(m["precision"][i]), 6)
            row[f"{name}_recall"] = round(float(m["recall"][i]), 6)
            # For a single true sign, within-sign accuracy is correct predictions
            # divided by the held-out clips for that sign; this equals recall.
            row[f"{name}_accuracy"] = round(float(m["recall"][i]), 6)
            row[f"{name}_f1"] = round(float(m["f1"][i]), 6)
            row[f"{name}_support"] = int(m["support"][i])
            per_class_time = detail["models"][name]["per_class_latency"][cname]  # type: ignore[index]
            row[f"{name}_latency_ms_mean"] = round(per_class_time["mean_ms"], 4)
            row[f"{name}_latency_samples"] = per_class_time["samples"]
        rows.append(row)
    per_class_path = out_dir / f"per_class_comparison{suffix}.csv"
    with open(per_class_path, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=header)
        wr.writeheader()
        wr.writerows(rows)

    # ---------------- summary_metrics.csv ----------------------------------
    fields = ["model", "protocol", "split", "n_samples", "n_classes", "parameters", "accuracy",
              "macro_precision", "macro_recall", "macro_f1",
              "weighted_precision", "weighted_recall", "weighted_f1",
              "latency_ms_mean", "latency_ms_p50", "latency_ms_p95",
              "latency_ms_batch_per_sample", "throughput_fps", "batch_throughput_fps"]
    summary_path = out_dir / f"summary_metrics{suffix}.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        for name in model_names:
            wr.writerow({k: (round(v, 6) if isinstance(v, float) else v)
                         for k, v in summary_rows[name].items() if k in fields})

    # ---------------- comparison plots --------------------------------------
    plot_model_comparison(summary_rows, results_dir / f"model_comparison{suffix}.png")
    plot_latency(summary_rows, results_dir / f"latency_comparison{suffix}.png")

    detail["ranking"] = sorted(
        ({"model": n, "accuracy": summary_rows[n]["accuracy"],
          "macro_f1": summary_rows[n]["macro_f1"],
          "weighted_f1": summary_rows[n]["weighted_f1"],
          "parameters": summary_rows[n]["parameters"],
          "latency_ms_mean": summary_rows[n]["latency_ms_mean"]} for n in model_names),
        key=lambda d: -d["macro_f1"],
    )
    detail["best_model_by_macro_f1"] = detail["ranking"][0]["model"] if detail["ranking"] else None
    detail["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    detail_path = results_dir / f"evaluation_summary_{protocol}.json"
    detail_path.write_text(json.dumps(detail, indent=2))

    return {"per_class_csv": per_class_path, "summary_csv": summary_path,
            "summary": summary_rows, "detail_json": detail_path,
            "ranking": detail["ranking"]}


def _print_report(res: Dict[str, object]) -> None:
    summary: Dict[str, Dict[str, object]] = res["summary"]  # type: ignore[assignment]
    print("\n=== Summary metrics (identical frozen test set) ===")
    print(f"{'model':<9}{'acc':>8}{'macroP':>9}{'macroR':>9}{'macroF1':>9}"
          f"{'wF1':>9}{'params':>10}{'lat ms':>9}{'FPS':>7}")
    for name, s in summary.items():
        print(f"{name:<9}{s['accuracy']:>8.4f}{s['macro_precision']:>9.4f}{s['macro_recall']:>9.4f}"
              f"{s['macro_f1']:>9.4f}{s['weighted_f1']:>9.4f}{s['parameters']:>10d}"
              f"{s['latency_ms_mean']:>9.2f}{s['throughput_fps']:>7.2f}")
    rank = res["ranking"]  # type: ignore[assignment]
    if rank:
        print(f"\nbest by macro-F1: {rank[0]['model']} "
              f"(macro-F1={rank[0]['macro_f1']:.4f}, acc={rank[0]['accuracy']:.4f})")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--protocol", default=config.SPLIT_PROTOCOL,
                    choices=("session_disjoint", "official", "hf"))
    ap.add_argument("--models", nargs="+", default=list(config.AVAILABLE_MODELS))
    ap.add_argument("--split", default="test", choices=("val", "test"))
    ap.add_argument("--results-dir", default=str(config.RESULTS_DIR))
    ap.add_argument("--suffix", default=None,
                    help="filename suffix; defaults to '' for the primary protocol "
                         "and '_<protocol>' for any other protocol")
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    suffix = args.suffix if args.suffix is not None else (
        "" if args.protocol == config.SPLIT_PROTOCOL else f"_{args.protocol}")

    res = evaluate(args.protocol, args.models, results_dir, results_dir,
                   suffix=suffix, repeats=args.repeats, split=args.split)
    _print_report(res)
    print(f"\nwrote {res['per_class_csv']}\nwrote {res['summary_csv']}\nwrote {res['detail_json']}")
    print(f"plots: confusion_*{suffix}.png, model_comparison{suffix}.png, latency_comparison{suffix}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
