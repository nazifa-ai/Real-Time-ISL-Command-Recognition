"""
Shared training machinery for the three models.

Guarantees enforced here (the fairness contract of the experiment):

* one seed for numpy / python / TensorFlow, re-applied per model;
* identical processed arrays (``training/preprocess.py`` output) for all models;
* identical augmentation (offline, seeded, train split only);
* identical callbacks: best-checkpoint, early stopping, LR reduction;
* identical epochs / batch size / loss / label smoothing;
* every model directory records the class mapping, the preprocessing config and
  the full history, so an evaluator can reproduce the run.
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

import config
from training import models as model_zoo
from training import preprocess as prep

LOG = logging.getLogger("train")


# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
def set_seeds(seed: int = config.RANDOM_SEED) -> None:
    """Seed every RNG that can influence a training run."""
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.keras.utils.set_random_seed(seed)
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:  # pragma: no cover - not supported on every build
            pass
        try:
            tf.config.threading.set_intra_op_parallelism_threads(config.NUM_THREADS)
            tf.config.threading.set_inter_op_parallelism_threads(max(1, config.NUM_THREADS // 2))
        except Exception:  # pragma: no cover
            pass
    except Exception as exc:  # pragma: no cover
        LOG.warning("TensorFlow seeding unavailable: %s", exc)


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
@dataclass
class TrainingData:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    classes: list
    summary: Dict

    def one_hot(self, y: np.ndarray, num_classes: int) -> np.ndarray:
        import tensorflow as tf

        return tf.keras.utils.to_categorical(y, num_classes=num_classes)


def load_training_data(protocol: Optional[str] = None) -> TrainingData:
    protocol = protocol or config.SPLIT_PROTOCOL
    data = prep.load_processed(protocol)
    summary_path = Path(config.PROCESSED_DIR) / f"preprocess_summary_{protocol}.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    classes = [str(c) for c in data["classes"].tolist()]
    return TrainingData(
        X_train=data["X_train"], y_train=data["y_train"],
        X_val=data["X_val"], y_val=data["y_val"],
        X_test=data["X_test"], y_test=data["y_test"],
        classes=classes, summary=summary,
    )


# --------------------------------------------------------------------------
# Callbacks / artefacts
# --------------------------------------------------------------------------
def build_callbacks(model_dir: Path):
    import tensorflow as tf

    model_dir.mkdir(parents=True, exist_ok=True)
    ckpt = model_dir / "best.keras"
    return [
        tf.keras.callbacks.ModelCheckpoint(str(ckpt), monitor="val_accuracy",
                                           save_best_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy",
                                         patience=config.EARLY_STOPPING_PATIENCE,
                                         restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",
                                             factor=config.REDUCE_LR_FACTOR,
                                             patience=config.REDUCE_LR_PATIENCE,
                                             min_lr=config.MIN_LEARNING_RATE, verbose=1),
    ], ckpt


def save_artifacts(model_name: str, model, history, data: TrainingData,
                   model_dir: Path, ckpt: Path, extra: Optional[Dict] = None) -> Dict:
    """Persist model, class mapping, preprocessing config and training history."""
    model_dir.mkdir(parents=True, exist_ok=True)
    final_path = model_dir / "model.keras"
    model.save(final_path)

    (model_dir / "class_mapping.json").write_text(json.dumps({
        "classes": data.classes,
        "index": {c: i for i, c in enumerate(data.classes)},
        "num_classes": len(data.classes),
        "source": "datasets/metadata/vocabulary.json (official INCLUDE-50 metadata)",
    }, indent=2))

    (model_dir / "preprocessing_config.json").write_text(json.dumps({
        "feature_dim": config.FEATURE_DIM,
        "model_input_dim": config.MODEL_INPUT_DIM,
        "mask_enabled": config.MASK_ENABLED,
        "sequence_length": config.SEQUENCE_LENGTH,
        "seq_resample_method": config.SEQ_RESAMPLE_METHOD,
        "min_valid_frames": config.MIN_VALID_FRAMES,
        "x_axis_scale": config.X_AXIS_SCALE,
        "axis_correction_applied": config.APPLY_KEYPOINT_AXIS_CORRECTION,
        "normalisation": data.summary.get("normalisation", {}),
        "augmentation": data.summary.get("augmentation", {}),
        "layout": {"pose": [0, 33], "left_hand": [33, 54], "right_hand": [54, 75]},
        "split_protocol": data.summary.get("protocol"),
    }, indent=2))

    hist = {k: [float(x) for x in v] for k, v in (history.history if history else {}).items()}
    (model_dir / "training_history.json").write_text(json.dumps({
        "model": model_name,
        "history": hist,
        "epochs_run": len(hist.get("loss", [])),
        "parameters": model_zoo.parameter_count(model),
        "best_val_accuracy": float(max(hist.get("val_accuracy", [0.0]))),
        "config": {
            "batch_size": config.BATCH_SIZE, "epochs": config.EPOCHS,
            "learning_rate": config.LEARNING_RATE, "dropout": config.DROPOUT,
            "label_smoothing": config.LABEL_SMOOTHING, "seed": config.RANDOM_SEED,
            "optimizer": "adam", "loss": "categorical_crossentropy",
        },
        **(extra or {}),
    }, indent=2))
    LOG.info("artifacts saved to %s (params=%d)", model_dir, model_zoo.parameter_count(model))
    return {"model": str(final_path), "checkpoint": str(ckpt)}


def plot_history(model_name: str, history, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h = history.history if history else {}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    if "loss" in h:
        axes[0].plot(h["loss"], label="train")
        axes[0].plot(h.get("val_loss", []), label="val")
        axes[0].set_title(f"{model_name}: loss")
        axes[0].set_xlabel("epoch")
        axes[0].legend()
        axes[0].grid(alpha=0.3)
    if "accuracy" in h:
        axes[1].plot(h["accuracy"], label="train")
        axes[1].plot(h.get("val_accuracy", []), label="val")
        axes[1].set_title(f"{model_name}: accuracy")
        axes[1].set_xlabel("epoch")
        axes[1].legend()
        axes[1].grid(alpha=0.3)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# Main entry used by train_lstm.py / train_gru.py / train_gru_mha.py
# --------------------------------------------------------------------------
def train(model_name: str, epochs: Optional[int] = None, batch_size: Optional[int] = None,
          protocol: Optional[str] = None, verbose: int = 2) -> Dict:
    epochs = int(epochs or config.EPOCHS)
    batch_size = int(batch_size or config.BATCH_SIZE)
    protocol = protocol or config.SPLIT_PROTOCOL

    set_seeds(config.RANDOM_SEED)
    data = load_training_data(protocol)
    num_classes = len(data.classes)
    input_dim = data.X_train.shape[-1]

    LOG.info("model=%s protocol=%s train=%s val=%s test=%s classes=%d input_dim=%d",
             model_name, protocol, data.X_train.shape, data.X_val.shape, data.X_test.shape,
             num_classes, input_dim)

    model = model_zoo.build_model(model_name, input_dim=input_dim, num_classes=num_classes)
    model.summary(print_fn=lambda s: LOG.info("  %s", s))

    model_dir = Path(config.MODEL_PATHS[model_name])
    callbacks, ckpt = build_callbacks(model_dir)

    y_train = data.one_hot(data.y_train, num_classes)
    y_val = data.one_hot(data.y_val, num_classes)

    t0 = time.time()
    history = model.fit(
        data.X_train, y_train,
        validation_data=(data.X_val, y_val),
        epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=verbose,
    )
    seconds = time.time() - t0

    paths = save_artifacts(model_name, model, history, data, model_dir, ckpt,
                           extra={"protocol": protocol, "train_seconds": round(seconds, 1),
                                  "train_samples": int(data.X_train.shape[0]),
                                  "val_samples": int(data.X_val.shape[0]),
                                  "test_samples": int(data.X_test.shape[0])})
    plot_history(model_name, history, Path(config.RESULTS_DIR) / f"training_history_{model_name}.png")

    val_loss, val_acc = model.evaluate(data.X_val, y_val, verbose=0)
    LOG.info("%s: finished in %.1fs | best val_accuracy=%.4f | final val_accuracy=%.4f",
             model_name, seconds, max(history.history.get("val_accuracy", [0.0])), val_acc)
    return {"model": model_name, "val_accuracy": float(val_acc), "seconds": seconds, **paths}


def run_cli(model_name: str, argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=f"Train the {model_name} model on INCLUDE-50")
    ap.add_argument("--epochs", type=int, default=config.EPOCHS)
    ap.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    ap.add_argument("--protocol", default=config.SPLIT_PROTOCOL,
                    choices=["session_disjoint", "official", "hf"])
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    train(model_name, epochs=args.epochs, batch_size=args.batch_size,
          protocol=args.protocol, verbose=1 if args.quiet else 2)
    return 0
