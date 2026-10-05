"""
Conservative personalisation: profiles, vocabulary subsets, per-user thresholds,
calibration samples and an *optional*, explicitly-flagged fine-tune.

Design rules (documented in ``docs/system_card.md`` and ``docs/limitations.md``):

* a profile never invents vocabulary - it can only *restrict* the 50 official
  INCLUDE-50 classes that the model was trained on;
* the per-user threshold is clamped to ``+-PERSONALIZATION_MAX_THRESHOLD_DELTA``
  around the global default, so a user cannot silently disable the confidence gate;
* calibration samples are stored as *features* (never video) - the raw upload is
  discarded unless ``PERSIST_UPLOADS`` is explicitly enabled;
* fine-tuning is opt-in, needs ``PERSONALIZATION_MIN_SAMPLES_FOR_FINETUNE`` samples,
  runs for a few epochs at a low learning rate and its reported accuracy is a
  *training-set* number unless enough samples exist for a held-out estimate.  The
  project makes no claim that personalisation improves real-world accuracy.

Layout::

    models/personalized/<profile>/profile.json      settings + history
    models/personalized/<profile>/calibration.npz   X (N,30,225), y (N,), labels
    models/personalized/<profile>/model.keras       optional fine-tuned weights
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import config

LOG = logging.getLogger("personalization")

PROFILE_ROOT = Path(config.MODELS_DIR) / "personalized"
_NAME_RE = re.compile(r"^[A-Za-z0-9_\-]{1,32}$")


class PersonalizationError(ValueError):
    """Raised for invalid profile names, vocabularies or sample shapes."""


# --------------------------------------------------------------------------
def _dir(name: str) -> Path:
    if not _NAME_RE.match(name or ""):
        raise PersonalizationError("profile name must match [A-Za-z0-9_-]{1,32}")
    return PROFILE_ROOT / name


def profile_dir(name: str) -> Path:
    return _dir(name)


def list_profiles() -> List[Dict[str, object]]:
    if not PROFILE_ROOT.exists():
        return []
    out = []
    for p in sorted(PROFILE_ROOT.iterdir()):
        if p.is_dir() and (p / "profile.json").exists():
            try:
                out.append(json.loads((p / "profile.json").read_text()))
            except json.JSONDecodeError:
                continue
    return out


def load_profile(name: str) -> Dict[str, object]:
    path = _dir(name) / "profile.json"
    if not path.exists():
        raise PersonalizationError(f"profile '{name}' does not exist")
    return json.loads(path.read_text())


def _save_profile(profile: Dict[str, object]) -> Dict[str, object]:
    d = _dir(str(profile["name"]))
    d.mkdir(parents=True, exist_ok=True)
    profile["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    (d / "profile.json").write_text(json.dumps(profile, indent=2))
    return profile


# --------------------------------------------------------------------------
def base_threshold() -> float:
    return float(config.CONFIDENCE_THRESHOLD)


def clamp_threshold(value: float) -> float:
    lo = base_threshold() - float(config.PERSONALIZATION_MAX_THRESHOLD_DELTA)
    hi = base_threshold() + float(config.PERSONALIZATION_MAX_THRESHOLD_DELTA)
    return float(min(max(float(value), max(0.01, lo)), min(0.999, hi)))


def _validate_vocabulary(vocab: Optional[Sequence[str]]) -> List[str]:
    if not vocab:
        return []
    from backend.preprocessing.include50 import load_vocabulary

    official = {c.upper(): c for c in load_vocabulary()}
    unknown = [v for v in vocab if str(v).strip().upper() not in official]
    if unknown:
        raise PersonalizationError(
            f"vocabulary entries must come from the official INCLUDE-50 list; unknown: {unknown[:5]}"
        )
    return [official[str(v).strip().upper()] for v in vocab]


def create_profile(name: str, base_model: Optional[str] = None,
                   vocabulary: Optional[Sequence[str]] = None,
                   threshold: Optional[float] = None,
                   preset: Optional[str] = None) -> Dict[str, object]:
    base_model = base_model or config.DEFAULT_MODEL
    if base_model not in config.AVAILABLE_MODELS:
        raise PersonalizationError(f"base_model must be one of {list(config.AVAILABLE_MODELS)}")

    vocab = list(vocabulary or [])
    if preset:
        preset_map = config.PERSONALIZATION_DEFAULT_VOCAB_PRESETS
        if preset not in preset_map:
            raise PersonalizationError(f"unknown preset '{preset}'; options: {list(preset_map)}")
        vocab = list(preset_map[preset])
    vocab = _validate_vocabulary(vocab)

    profile = {
        "name": name,
        "base_model": base_model,
        "vocabulary": vocab,                 # [] -> full 50-class vocabulary
        "threshold": clamp_threshold(threshold if threshold is not None else base_threshold()),
        "threshold_requested": threshold,
        "calibration_samples": 0,
        "per_class_samples": {},
        "fine_tuned": False,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "notes": "Personalisation is opt-in and its benefit is not proven; see docs/limitations.md.",
    }
    return _save_profile(profile)


def update_profile(name: str, vocabulary: Optional[Sequence[str]] = None,
                   threshold: Optional[float] = None,
                   base_model: Optional[str] = None) -> Dict[str, object]:
    profile = load_profile(name)
    if vocabulary is not None:
        profile["vocabulary"] = _validate_vocabulary(vocabulary)
    if threshold is not None:
        profile["threshold_requested"] = float(threshold)
        profile["threshold"] = clamp_threshold(float(threshold))
    if base_model is not None:
        if base_model not in config.AVAILABLE_MODELS:
            raise PersonalizationError(f"base_model must be one of {list(config.AVAILABLE_MODELS)}")
        profile["base_model"] = base_model
    return _save_profile(profile)


def delete_profile(name: str) -> bool:
    d = _dir(name)
    if not d.exists():
        return False
    import shutil

    shutil.rmtree(d)
    return True


# --------------------------------------------------------------------------
# calibration samples
# --------------------------------------------------------------------------
def _npz_path(name: str) -> Path:
    return _dir(name) / "calibration.npz"


def load_calibration(name: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    p = _npz_path(name)
    if not p.exists():
        return None, None
    data = np.load(p, allow_pickle=True)
    return data["X"], data["y"]


def add_calibration_samples(name: str, features: np.ndarray, labels: Sequence[str]) -> Dict[str, object]:
    """Append confirmed clips (features only) to a profile's calibration set."""
    profile = load_profile(name)
    from backend.preprocessing.include50 import load_vocabulary

    official = {c.upper(): c for c in load_vocabulary()}
    X = np.asarray(features, dtype=np.float32)
    if X.ndim == 2:
        X = X[None, ...]
    if X.ndim != 3 or X.shape[-1] != config.MODEL_INPUT_DIM:
        raise PersonalizationError(
            f"features must have shape (N, T, {config.MODEL_INPUT_DIM}); got {X.shape}"
        )
    if len(labels) != X.shape[0]:
        raise PersonalizationError("one label per sample is required")

    y_idx: List[int] = []
    classes = compile_vocabulary(profile)
    for lab in labels:
        key = str(lab).strip().upper()
        if key not in official:
            raise PersonalizationError(f"label '{lab}' is not part of INCLUDE-50")
        if key not in {c.upper() for c in classes}:
            raise PersonalizationError(
                f"label '{lab}' is outside this profile's vocabulary {profile['vocabulary']}"
            )
        y_idx.append(next(i for i, c in enumerate(classes) if c.upper() == key))

    old_X, old_y = load_calibration(name)
    if old_X is not None and old_y is not None and old_X.shape[0]:
        if old_X.shape[-1] != X.shape[-1]:
            raise PersonalizationError("stored calibration samples use a different feature layout")
        X = np.concatenate([old_X, X], axis=0)
        y = np.concatenate([old_y.astype(np.int64), np.array(y_idx, dtype=np.int64)])
    else:
        y = np.array(y_idx, dtype=np.int64)

    d = _dir(name)
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(_npz_path(name), X=X, y=y, classes=np.array(classes, dtype=object))

    counts: Dict[str, int] = {}
    for i in y:
        counts[classes[int(i)]] = counts.get(classes[int(i)], 0) + 1
    profile["calibration_samples"] = int(len(y))
    profile["per_class_samples"] = counts
    return _save_profile(profile)


def compile_vocabulary(profile: Dict[str, object]) -> List[str]:
    """Full 50-class list when the profile has no subset, else the subset (index space preserved)."""
    from backend.preprocessing.include50 import load_vocabulary

    all_classes = load_vocabulary()
    subset = profile.get("vocabulary") or []
    if not subset:
        return list(all_classes)
    keep = {str(c).upper() for c in subset}          # type: ignore[union-attr]
    return [c for c in all_classes if c.upper() in keep]


# --------------------------------------------------------------------------
# fine-tuning (optional)
# --------------------------------------------------------------------------
def can_fine_tune(name: str) -> Tuple[bool, str]:
    profile = load_profile(name)
    n = int(profile.get("calibration_samples") or 0)
    need = int(config.PERSONALIZATION_MIN_SAMPLES_FOR_FINETUNE)
    if not config.PERSONALIZATION_ENABLED:
        return False, "personalisation is disabled (PERSONALIZATION_ENABLED=0)"
    if n < need:
        return False, f"need at least {need} calibration samples, have {n}"
    X, y = load_calibration(name)
    if X is None or len(np.unique(y)) < 2:
        return False, "at least two distinct classes are required"
    return True, "ok"


def fine_tune(name: str, epochs: Optional[int] = None, seed: int = config.RANDOM_SEED,
              val_fraction: float = 0.2) -> Dict[str, object]:
    """Adapt a copy of the base model to one user's calibration samples.

    Only the copy under ``models/personalized/<name>/`` is modified, so the
    scientific comparison in ``results/`` stays uncontaminated.
    """
    ok, why = can_fine_tune(name)
    if not ok:
        raise PersonalizationError(f"cannot fine-tune profile '{name}': {why}")

    import tensorflow as tf

    from training import preprocess as prep
    from training.models import build_model

    profile = load_profile(name)
    classes = compile_vocabulary(profile)
    X, y = load_calibration(name)
    assert X is not None and y is not None

    # map stored indices (profile index space) to global class indices used by the model
    from backend.preprocessing.include50 import load_vocabulary

    global_classes = load_vocabulary()
    global_index = {c: i for i, c in enumerate(global_classes)}
    y_global = np.array([global_index[classes[int(i)]] for i in y], dtype=np.int64)

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(y_global))
    X, y_global = X[order], y_global[order]
    n_val = int(round(len(y_global) * val_fraction))
    if len(y_global) - n_val < 2:
        n_val = 0
    X_train, y_train = X[: len(X) - n_val], y_global[: len(y_global) - n_val]
    X_val, y_val = X[len(X) - n_val:], y_global[len(y_global) - n_val:]

    base_path = Path(config.MODEL_PATHS[str(profile["base_model"])]) / "model.keras"
    if not base_path.exists():
        raise PersonalizationError(f"base model missing: {base_path}")
    model = tf.keras.models.load_model(str(base_path))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=float(config.PERSONALIZATION_FINETUNE_LR)),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    num_classes = len(global_classes)
    hist = model.fit(
        X_train.astype(np.float32),
        tf.keras.utils.to_categorical(y_train, num_classes),
        validation_data=((X_val.astype(np.float32), tf.keras.utils.to_categorical(y_val, num_classes))
                         if n_val else None),
        epochs=int(epochs or config.PERSONALIZATION_FINETUNE_EPOCHS),
        batch_size=max(1, min(config.BATCH_SIZE, len(X_train))),
        verbose=0,
    )

    d = _dir(name)
    d.mkdir(parents=True, exist_ok=True)
    model.save(d / "model.keras")

    train_acc = float(hist.history["accuracy"][-1])
    val_acc = float(hist.history["val_accuracy"][-1]) if "val_accuracy" in hist.history else None
    result = {
        "profile": name,
        "samples_used": int(len(X)),
        "train_samples": int(len(X_train)),
        "heldout_samples": int(len(X_val)),
        "epochs": int(epochs or config.PERSONALIZATION_FINETUNE_EPOCHS),
        "learning_rate": float(config.PERSONALIZATION_FINETUNE_LR),
        "train_accuracy_on_calibration": round(train_acc, 4),
        "heldout_accuracy": (round(val_acc, 4) if val_acc is not None else None),
        "caveat": ("train accuracy on a handful of self-recorded clips is NOT evidence of "
                   "real-world improvement; see docs/limitations.md"),
        "model": str(d / "model.keras"),
    }
    profile["fine_tuned"] = True
    profile["fine_tune_history"] = (list(profile.get("fine_tune_history") or []) + [result])[-5:]
    _save_profile(profile)
    return result


def reset_fine_tune(name: str) -> Dict[str, object]:
    profile = load_profile(name)
    p = _dir(name) / "model.keras"
    if p.exists():
        p.unlink()
    profile["fine_tuned"] = False
    return _save_profile(profile)


# --------------------------------------------------------------------------
# applying a profile to a live engine
# --------------------------------------------------------------------------
def apply_profile_to_engine(engine, name: str) -> Dict[str, object]:
    """Set the vocabulary subset and per-user threshold on a ``PredictionEngine``."""
    profile = load_profile(name)
    kept = engine.set_vocab_filter(profile["vocabulary"] or None)
    engine.threshold = clamp_threshold(float(profile["threshold"]))
    engine.smoother.reset()
    return {
        "profile": name,
        "vocabulary_size": len(kept) if kept else len(engine.classes),
        "vocabulary_filter": kept or None,
        "threshold": engine.threshold,
        "fine_tuned_weights": str(_dir(name) / "model.keras"),
    }
