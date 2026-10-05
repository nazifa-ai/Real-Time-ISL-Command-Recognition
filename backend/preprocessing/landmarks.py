"""
Canonical landmark representation, geometric normalisation and missing-landmark
handling.

Design contract (used by preprocessing, training, inference and the robustness
experiments - all of them import from this module and nothing else):

    frame  ->  np.ndarray[float32] of shape (FEATURE_DIM,)   [225 by default]
               optional mask of shape (NUM_LANDMARKS,) appended when
               ``config.MASK_ENABLED`` is True.

Layout
------
    slots   0..32  : pose            (MediaPipe BlazePose, 33 points)
    slots  33..53  : left hand       (MediaPipe Hands, 21 points)
    slots  54..74  : right hand      (MediaPipe Hands, 21 points)

Missing landmarks are *never* silently invented: coordinates are set to 0.0 and
the corresponding mask slot is set to 0.0 so a model can learn to ignore them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

import config

POSE_SLICE = config.LAYOUT.slice("pose")
LEFT_SLICE = config.LAYOUT.slice("left_hand")
RIGHT_SLICE = config.LAYOUT.slice("right_hand")

COORDS = config.COORDS_PER_LANDMARK
NUM_LM = config.NUM_LANDMARKS


# --------------------------------------------------------------------------
# Containers
# --------------------------------------------------------------------------
@dataclass
class FrameLandmarks:
    """One video frame worth of MediaPipe output (already in canonical slots)."""

    pose: Optional[np.ndarray] = None            # (33, 3) or (33, 4) with visibility
    left_hand: Optional[np.ndarray] = None       # (21, 3)
    right_hand: Optional[np.ndarray] = None      # (21, 3)
    pose_visibility: Optional[np.ndarray] = None # (33,)

    def is_empty(self) -> bool:
        return self.pose is None and self.left_hand is None and self.right_hand is None


def empty_landmarks() -> np.ndarray:
    """(75, 3) zero array - the representation of 'nothing detected'."""
    return np.zeros((NUM_LM, COORDS), dtype=np.float32)


def mask_from_coords(coords: np.ndarray, visibility: Optional[np.ndarray] = None) -> np.ndarray:
    """(75,) presence mask.

    A landmark counts as present when either it carries explicit visibility above
    ``config.VISIBILITY_THRESHOLD`` or its coordinates are non-zero.
    """
    coords = np.asarray(coords, dtype=np.float32)
    present = np.any(np.abs(coords) > 0.0, axis=-1)
    if visibility is not None:
        vis = np.asarray(visibility, dtype=np.float32)
        if vis.shape[0] == coords.shape[0]:
            present = present & (vis >= config.VISIBILITY_THRESHOLD)
    return present.astype(np.float32)


# --------------------------------------------------------------------------
# Packing / unpacking
# --------------------------------------------------------------------------
def pack_frame(frame: FrameLandmarks) -> Tuple[np.ndarray, np.ndarray]:
    """Canonicalise one frame.

    Returns
    -------
    coords : (75, 3) float32
    mask   : (75,)   float32
    """
    coords = empty_landmarks()
    mask = np.zeros((NUM_LM,), dtype=np.float32)

    if frame.pose is not None and len(frame.pose):
        pose_raw = np.asarray(frame.pose, dtype=np.float32)
        pose = pose_raw[:, :COORDS]
        n = min(pose.shape[0], POSE_SLICE.stop - POSE_SLICE.start)
        coords[POSE_SLICE.start: POSE_SLICE.start + n] = pose[:n]
        vis = None
        if frame.pose_visibility is not None:
            vis = np.asarray(frame.pose_visibility, dtype=np.float32)[:n]
        elif pose_raw.shape[-1] > COORDS:
            vis = pose_raw[:n, COORDS]
        mask[POSE_SLICE.start: POSE_SLICE.start + n] = mask_from_coords(
            coords[POSE_SLICE.start: POSE_SLICE.start + n], vis
        )

    for hand, sl in ((frame.left_hand, LEFT_SLICE), (frame.right_hand, RIGHT_SLICE)):
        if hand is None:
            continue
        hand = np.asarray(hand, dtype=np.float32)[:, :COORDS]
        n = min(hand.shape[0], sl.stop - sl.start)
        coords[sl.start: sl.start + n] = hand[:n]
        mask[sl.start: sl.start + n] = mask_from_coords(coords[sl.start: sl.start + n])

    # The training keypoint loader removes low-confidence landmarks before
    # normalisation. Apply the same rule to live MediaPipe output here so a
    # masked point cannot still affect body-centering or model features.
    coords[mask <= 0] = 0.0
    return coords, mask


def feature_vector(coords: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """(75,3) [+ (75,)] -> flat feature vector (225 by default, 300 with mask)."""
    flat = np.asarray(coords, dtype=np.float32).reshape(-1)
    if config.MASK_ENABLED:
        m = np.zeros((NUM_LM,), dtype=np.float32) if mask is None else np.asarray(mask, dtype=np.float32)
        return np.concatenate([flat, m]).astype(np.float32)
    return flat.astype(np.float32)


def unpack_features(vec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Inverse of :func:`feature_vector`."""
    vec = np.asarray(vec, dtype=np.float32)
    coords = vec[: config.FEATURE_DIM].reshape(NUM_LM, COORDS)
    if config.MASK_ENABLED and vec.size >= config.MODEL_INPUT_DIM:
        mask = vec[config.FEATURE_DIM: config.MODEL_INPUT_DIM]
    else:
        mask = mask_from_coords(coords)
    return coords, mask


# --------------------------------------------------------------------------
# Normalisation
# --------------------------------------------------------------------------
def reference_points(coords: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, float]:
    """Body reference = mid-shoulder, scale = shoulder width (fallback: torso/1.0)."""
    ls = coords[POSE_SLICE.start + config.POSE_LEFT_SHOULDER]
    rs = coords[POSE_SLICE.start + config.POSE_RIGHT_SHOULDER]
    have = (
        mask[POSE_SLICE.start + config.POSE_LEFT_SHOULDER] > 0
        and mask[POSE_SLICE.start + config.POSE_RIGHT_SHOULDER] > 0
    )
    if have:
        centre = (ls + rs) / 2.0
        scale = float(np.linalg.norm(rs - ls))
        if scale > 1e-6:
            return centre, scale

    # fallback 1: torso (shoulder-to-hip) using whichever side is available
    lh = coords[POSE_SLICE.start + config.POSE_LEFT_HIP]
    rh = coords[POSE_SLICE.start + config.POSE_RIGHT_HIP]
    if mask[POSE_SLICE.start + config.POSE_LEFT_HIP] > 0 and mask[POSE_SLICE.start + config.POSE_RIGHT_HIP] > 0:
        centre = (lh + rh) / 2.0
        scale = float(np.linalg.norm(lh - rh))
        if scale > 1e-6:
            return centre, scale

    # fallback 2: centroid of every present landmark, unit scale
    present = mask > 0
    if present.any():
        return coords[present].mean(axis=0), 1.0
    return np.zeros((COORDS,), dtype=np.float32), 1.0


def normalise(coords: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Translate to mid-shoulder and scale by shoulder width.

    Normalisation is a pure function of the *frame itself* (no temporal statistics,
    no cross-split statistics) so it introduces no train/test leakage.
    """
    coords = np.asarray(coords, dtype=np.float32)
    mask = np.asarray(mask, dtype=np.float32)
    if not (mask > 0).any():
        return np.zeros_like(coords)
    centre, scale = reference_points(coords, mask)
    out = (coords - centre[None, :]) / max(scale, 1e-6)
    out[mask <= 0] = 0.0
    return out.astype(np.float32)


def normalise_sequence(seq: np.ndarray, masks: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Normalise a whole clip.

    Parameters
    ----------
    seq : (T, 75, 3) or (T, 225) raw packed features

    Returns
    -------
    (T, 225[, +75]) normalised features, (T, 75) masks
    """
    seq = np.asarray(seq, dtype=np.float32)
    if seq.ndim == 2 and seq.shape[-1] == config.FEATURE_DIM:
        seq = seq.reshape(seq.shape[0], NUM_LM, COORDS)

    t = seq.shape[0]
    out = np.zeros((t, NUM_LM, COORDS), dtype=np.float32)
    output_masks = np.zeros((t, NUM_LM), dtype=np.float32)
    provided_masks = None if masks is None else np.asarray(masks, dtype=np.float32)
    if provided_masks is not None and provided_masks.shape != (t, NUM_LM):
        raise ValueError(f"masks must have shape {(t, NUM_LM)}, got {provided_masks.shape}")

    for i in range(t):
        m = (mask_from_coords(seq[i]) if provided_masks is None
             else (provided_masks[i] > 0).astype(np.float32))
        output_masks[i] = m
        visible = np.asarray(seq[i], dtype=np.float32).copy()
        visible[m <= 0] = 0.0
        out[i] = normalise(visible, m)

    feats = np.stack([feature_vector(out[i], output_masks[i]) for i in range(t)])
    return feats.astype(np.float32), output_masks


# --------------------------------------------------------------------------
# Geometric helpers used by the mirror-aware augmentation
# --------------------------------------------------------------------------
MIRROR_GROUPS = ((LEFT_SLICE.start, RIGHT_SLICE.start),)


def mirror_hands(coords: np.ndarray) -> np.ndarray:
    """Swap left/right hand slots and flip x.  Used only by augmentation."""
    out = coords.copy()
    left = coords[LEFT_SLICE.start: LEFT_SLICE.stop].copy()
    right = coords[RIGHT_SLICE.start: RIGHT_SLICE.stop].copy()
    out[LEFT_SLICE.start: LEFT_SLICE.stop] = right
    out[RIGHT_SLICE.start: RIGHT_SLICE.stop] = left
    out[:, 0] *= -1.0
    return out
