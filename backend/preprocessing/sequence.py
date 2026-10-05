"""
Temporal sequence construction and augmentation.

Everything that changes the time axis lives here so that training, evaluation and
the robustness experiments apply *exactly the same* operations.

Public API
----------
resample_sequence(seq, length, method)       fixed-length resampling (no leakage)
pad_or_crop(seq, length, mode)               deterministic padding / centre crop
make_windows(seq, length, stride)            sliding windows for long clips
temporal_crop(seq, frac, rng)                augmentation: random sub-clip
frame_drop(seq, prob, rng)                   augmentation *and* robustness op
landmark_dropout(seq, prob, rng)             augmentation *and* robustness op
gaussian_jitter(seq, sigma, rng)             augmentation *and* robustness op
frame_skip(seq, keep_ratio)                  deterministic frame dropping (robustness)
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

import config

COORDS_PER_FRAME = config.NUM_LANDMARKS * config.COORDS_PER_LANDMARK


def _as_2d(seq: np.ndarray) -> Tuple[np.ndarray, int, int]:
    """Return (seq_2d (T, D), T, D).  Accepts (T, 75, 3) or (T, D).

    A batched array ``(N, T, D)`` is rejected loudly: these operators work on one
    clip at a time, and silently reshaping a batch would corrupt the time axis.
    """
    seq = np.asarray(seq, dtype=np.float32)
    if seq.ndim == 1:
        seq = seq[None, :]
    elif seq.ndim == 3:
        if seq.shape[1] == config.NUM_LANDMARKS and seq.shape[2] == config.COORDS_PER_LANDMARK:
            seq = seq.reshape(seq.shape[0], -1)          # (T, 75, 3) -> (T, 225)
        else:
            raise ValueError(
                f"sequence operators take a single clip shaped (T, {COORDS_PER_FRAME}) or "
                f"(T, {config.NUM_LANDMARKS}, {config.COORDS_PER_LANDMARK}); got {seq.shape}. "
                "Loop over the batch instead."
            )
    elif seq.ndim > 3:
        raise ValueError(f"unsupported sequence shape {seq.shape}")
    if seq.ndim != 2:
        raise ValueError(f"unsupported sequence shape {seq.shape}")
    return seq, seq.shape[0], seq.shape[1]


def resample_sequence(seq: np.ndarray, length: int | None = None, method: str | None = None) -> np.ndarray:
    """Resample a clip to exactly ``length`` frames.

    ``linear``  : linear interpolation over normalised time (default, preserves
                  gesture dynamics; used for both training and evaluation).
    ``pad``     : deterministic edge padding / centre crop.
    ``uniform`` : pick ``length`` evenly spaced existing frames (no interpolation).
    """
    length = int(length or config.SEQUENCE_LENGTH)
    method = (method or config.SEQ_RESAMPLE_METHOD).lower()
    seq, t, _ = _as_2d(seq)

    if t == 0:
        return np.zeros((length, seq.shape[1]), dtype=np.float32)
    if t == length:
        return seq.astype(np.float32)

    src = np.linspace(0.0, 1.0, num=t, dtype=np.float64)
    dst = np.linspace(0.0, 1.0, num=length, dtype=np.float64)

    if method == "pad":
        return pad_or_crop(seq, length, mode=config.SEQ_PAD_MODE)
    if method == "uniform":
        idx = np.unique(np.round(dst * (t - 1)).astype(int))
        if idx.size < length:  # duplicate edges when the clip is very short
            idx = np.concatenate([idx, np.repeat(idx[-1], length - idx.size)])
        return seq[idx].astype(np.float32)

    out = np.empty((length, seq.shape[1]), dtype=np.float32)
    for c in range(seq.shape[1]):
        out[:, c] = np.interp(dst, src, seq[:, c].astype(np.float64)).astype(np.float32)
    return out


def pad_or_crop(seq: np.ndarray, length: int | None = None, mode: str = "edge") -> np.ndarray:
    """Deterministic fixed-length conversion: edge/zero pad, or centre crop."""
    length = int(length or config.SEQUENCE_LENGTH)
    seq, t, d = _as_2d(seq)
    if t == length:
        return seq.astype(np.float32)
    if t > length:
        start = (t - length) // 2
        return seq[start: start + length].astype(np.float32)
    pad = length - t
    before, after = pad // 2, pad - pad // 2
    if mode == "zero":
        filler = np.zeros((1, d), dtype=np.float32)
    else:  # 'edge'
        filler = seq[-1:]
    return np.concatenate([filler.repeat(before, axis=0), seq,
                           filler.repeat(after, axis=0)]).astype(np.float32)


def make_windows(seq: np.ndarray, length: int | None = None, stride: int | None = None) -> List[np.ndarray]:
    """Sliding windows (used by the real-time buffer and long-video batch mode)."""
    length = int(length or config.SEQUENCE_LENGTH)
    stride = int(stride or length)
    seq, t, _ = _as_2d(seq)
    if t < length:
        return [resample_sequence(seq, length)]
    return [seq[i: i + length] for i in range(0, t - length + 1, stride)]


# --------------------------------------------------------------------------
# Augmentations (training) / perturbations (robustness) - same implementations
# --------------------------------------------------------------------------
def temporal_crop(seq: np.ndarray, frac: float, rng: np.random.Generator) -> np.ndarray:
    """Keep a random sub-window (fraction of the clip), then the caller resamples."""
    seq, t, _ = _as_2d(seq)
    if frac <= 0 or t <= 2:
        return seq
    keep = max(2, int(round(t * (1.0 - frac))))
    start = int(rng.integers(0, t - keep + 1))
    return seq[start: start + keep]


def frame_drop(seq: np.ndarray, prob: float, rng: np.random.Generator) -> np.ndarray:
    """Randomly repeat neighbouring frames (simulates dropped/duplicated frames).

    The clip keeps its length, so downstream shapes never change.
    """
    seq, t, d = _as_2d(seq)
    if prob <= 0 or t < 3:
        return seq
    mask = rng.random(t) < prob
    mask[0] = False
    out = seq.copy()
    idx = np.nonzero(mask)[0]
    out[idx] = seq[idx - 1]
    return out


def frame_skip(seq: np.ndarray, keep_ratio: float) -> np.ndarray:
    """Deterministic frame dropping: keep every k-th frame (robustness protocol)."""
    seq, t, _ = _as_2d(seq)
    if keep_ratio >= 1.0 or t < 3:
        return seq
    step = max(2, int(round(1.0 / max(keep_ratio, 1e-6))))
    kept = seq[::step]
    if kept.shape[0] < 2:
        kept = seq[[0, -1]]
    return kept


def landmark_dropout(seq: np.ndarray, prob: float, rng: np.random.Generator,
                     groups: Tuple[str, ...] = ("left_hand", "right_hand", "pose")) -> np.ndarray:
    """Zero-out whole landmark groups (hand occlusion) with probability ``prob``."""
    seq, t, d = _as_2d(seq)
    if prob <= 0:
        return seq
    seq = seq.copy()
    for i in range(t):
        for g in groups:
            if rng.random() < prob:
                sl = config.LAYOUT.slice(g)
                lo, hi = sl.start * config.COORDS_PER_LANDMARK, sl.stop * config.COORDS_PER_LANDMARK
                if hi <= d:
                    seq[i, lo:hi] = 0.0
    return seq


def gaussian_jitter(seq: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Additive Gaussian noise on coordinates (mask channels are left untouched)."""
    seq, t, d = _as_2d(seq)
    if sigma <= 0:
        return seq
    noise = rng.normal(0.0, sigma, size=(t, config.FEATURE_DIM)).astype(np.float32)
    out = seq.copy()
    out[:, : config.FEATURE_DIM] += noise
    return out


def augment(seq: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Full training augmentation policy (identical for LSTM / GRU / GRU+MHA)."""
    cfg = config.AUGMENTATION
    if not cfg.get("enabled", True):
        return seq
    out = _as_2d(seq)[0]
    if cfg.get("temporal_crop_frac", 0) > 0:
        out = temporal_crop(out, float(cfg["temporal_crop_frac"]), rng)
        out = resample_sequence(out, config.SEQUENCE_LENGTH)
    if cfg.get("frame_drop_prob", 0) > 0:
        out = frame_drop(out, float(cfg["frame_drop_prob"]), rng)
    if cfg.get("landmark_dropout_prob", 0) > 0:
        out = landmark_dropout(out, float(cfg["landmark_dropout_prob"]), rng)
    if cfg.get("jitter_sigma", 0) > 0:
        out = gaussian_jitter(out, float(cfg["jitter_sigma"]), rng)
    if cfg.get("mirror_prob", 0) > 0 and rng.random() < float(cfg["mirror_prob"]):
        from .landmarks import mirror_hands

        out = out.reshape(out.shape[0], config.NUM_LANDMARKS, config.COORDS_PER_LANDMARK)
        out = np.stack([mirror_hands(f) for f in out]).reshape(out.shape[0], -1)
    return resample_sequence(out, config.SEQUENCE_LENGTH)
