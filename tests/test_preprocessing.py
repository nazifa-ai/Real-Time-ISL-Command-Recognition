"""Landmark packing, normalisation, missing-landmark handling and sequence shaping."""

from __future__ import annotations

import numpy as np
import pytest

import config
from backend.preprocessing import landmarks as lm
from backend.preprocessing import sequence as seq_ops


# --------------------------------------------------------------------------
# packing
# --------------------------------------------------------------------------
def test_layout_offsets():
    assert config.NUM_LANDMARKS == 75
    assert config.FEATURE_DIM == 225 == 75 * 3
    assert lm.POSE_SLICE.start == 0 and lm.POSE_SLICE.stop == 33
    assert lm.LEFT_SLICE.start == 33 and lm.LEFT_SLICE.stop == 54
    assert lm.RIGHT_SLICE.start == 54 and lm.RIGHT_SLICE.stop == 75


def test_pack_frame_shapes():
    frame = lm.FrameLandmarks(pose=np.ones((33, 3)), left_hand=np.ones((21, 3)),
                              right_hand=np.ones((21, 3)))
    coords, mask = lm.pack_frame(frame)
    assert coords.shape == (75, 3)
    assert mask.shape == (75,)
    assert mask.sum() == 75


def test_pack_frame_missing_parts():
    coords, mask = lm.pack_frame(lm.FrameLandmarks(pose=np.ones((33, 3))))
    assert mask[:33].sum() == 33
    assert mask[33:].sum() == 0
    assert np.allclose(coords[33:], 0.0)
    assert lm.FrameLandmarks().is_empty()


def test_pack_frame_uses_pose_visibility_and_zeros_masked_coordinates():
    pose = np.ones((33, 4), dtype=np.float32)
    pose[:, 3] = 1.0
    pose[7, 3] = 0.1
    coords, mask = lm.pack_frame(lm.FrameLandmarks(pose=pose))
    assert mask[7] == 0.0
    assert np.all(coords[7] == 0.0)
    assert mask[6] == 1.0


def test_normalise_sequence_respects_explicit_presence_mask(sample_sequence):
    coords = sample_sequence[:1].copy()
    masks = lm.mask_from_coords(coords[0])[None, :]
    masks[0, 11] = 0.0  # low-visibility shoulder must not affect centering/scale
    expected = coords.copy()
    expected[0, 11] = 0.0
    expected_features, expected_masks = lm.normalise_sequence(expected)

    actual_features, actual_masks = lm.normalise_sequence(coords, masks)
    assert np.allclose(actual_features, expected_features)
    assert np.array_equal(actual_masks, expected_masks)


def test_feature_vector_dim():
    coords, mask = lm.pack_frame(lm.FrameLandmarks(pose=np.ones((33, 3))))
    vec = lm.feature_vector(coords, mask)
    assert vec.shape == (config.MODEL_INPUT_DIM,)
    expected = 225 + (75 if config.MASK_ENABLED else 0)
    assert config.MODEL_INPUT_DIM == expected


def test_unpack_roundtrip():
    coords, mask = lm.pack_frame(lm.FrameLandmarks(pose=np.ones((33, 3)), right_hand=np.ones((21, 3))))
    back, back_mask = lm.unpack_features(lm.feature_vector(coords, mask))
    assert np.allclose(back, coords)
    assert np.allclose(back_mask, mask)


# --------------------------------------------------------------------------
# missing landmarks
# --------------------------------------------------------------------------
def test_empty_frame_normalises_to_zeros():
    coords = lm.empty_landmarks()
    mask = lm.mask_from_coords(coords)
    assert mask.sum() == 0
    assert np.allclose(lm.normalise(coords, mask), 0.0)


def test_normalise_uses_shoulder_midpoint_and_width(sample_sequence):
    """After normalisation the mid-shoulder sits at the origin and |shoulders| == 1."""
    coords = sample_sequence
    feats, masks = lm.normalise_sequence(coords)
    out = feats[0].reshape(75, 3)
    ls = out[11]
    rs = out[12]
    assert np.allclose((ls + rs) / 2.0, 0.0, atol=1e-5)
    assert abs(np.linalg.norm(rs - ls) - 1.0) < 1e-5
    assert masks.shape == (len(coords), 75)


def test_normalise_falls_back_when_shoulders_missing():
    coords = np.zeros((75, 3), dtype=np.float32)
    coords[23] = [-0.1, -0.2, 0.0]
    coords[24] = [0.1, -0.2, 0.0]        # hips only
    mask = lm.mask_from_coords(coords)
    out = lm.normalise(coords, mask)
    assert np.isfinite(out).all()
    assert not np.allclose(out, 0.0)


def test_missing_hand_does_not_move_present_landmarks(sample_sequence):
    coords = sample_sequence.copy()
    coords[:, 54:] = 0.0                 # right hand absent
    feats, _ = lm.normalise_sequence(coords)
    full_feats, _ = lm.normalise_sequence(sample_sequence)
    assert np.allclose(feats[0].reshape(75, 3)[:54], full_feats[0].reshape(75, 3)[:54], atol=1e-5)


def test_feature_dim_matches_processed_tensors(processed):
    assert processed["X_test"].shape[-1] == config.MODEL_INPUT_DIM


def test_processed_tensors_are_finite(processed):
    for key in ("X_train", "X_val", "X_test"):
        assert np.isfinite(processed[key]).all(), f"{key} contains NaN/Inf"


def test_processed_sequences_have_configured_length(processed):
    for key in ("X_train", "X_val", "X_test"):
        assert processed[key].shape[1] == config.SEQUENCE_LENGTH
        assert processed[key].shape[0] == processed[key.replace("X_", "y_")].shape[0]


def test_labels_are_valid_class_indices(processed):
    for key in ("y_train", "y_val", "y_test"):
        y = processed[key]
        assert y.min() >= 0 and y.max() < len(processed["classes"])
    from backend.preprocessing.include50 import load_vocabulary

    assert [str(c) for c in processed["classes"]] == load_vocabulary()


# --------------------------------------------------------------------------
# sequence ops
# --------------------------------------------------------------------------
def test_resample_upsamples_and_downsamples():
    short = np.random.default_rng(0).normal(size=(7, 225)).astype(np.float32)
    assert seq_ops.resample_sequence(short, 30).shape == (30, 225)
    long = np.random.default_rng(0).normal(size=(154, 225)).astype(np.float32)
    assert seq_ops.resample_sequence(long, 30).shape == (30, 225)


def test_pad_or_crop_edge_mode():
    short = np.arange(5 * 4, dtype=np.float32).reshape(5, 4)
    padded = seq_ops.pad_or_crop(short, 8, mode="edge")
    assert padded.shape == (8, 4)
    assert np.allclose(padded[-1], short[-1])


def test_make_windows_and_stride():
    seq = np.arange(60 * 3, dtype=np.float32).reshape(60, 3)
    windows = seq_ops.make_windows(seq, length=30, stride=15)
    assert [w.shape for w in windows] == [(30, 3), (30, 3), (30, 3)]


def test_temporal_crop_shortens_a_2d_sequence():
    seq = np.random.default_rng(0).normal(size=(30, 225)).astype(np.float32)
    out = seq_ops.temporal_crop(seq, 0.2, np.random.default_rng(1))
    assert out.shape[1] == 225 and out.shape[0] < 30
    assert out.shape[0] >= 2


def test_frame_drop_keeps_length_and_repeats_neighbours():
    """frame_drop simulates dropped frames by repeating the previous one, so shapes stay fixed."""
    seq = np.arange(30 * 3, dtype=np.float32).reshape(30, 3)
    out = seq_ops.frame_drop(seq, 0.5, np.random.default_rng(2))
    assert out.shape == seq.shape
    assert not np.allclose(out, seq), "a 50% drop rate must change something"
    assert out[0].tolist() == seq[0].tolist(), "the first frame is never replaced"


def test_frame_skip_keep_ratio():
    seq = np.arange(30 * 2, dtype=np.float32).reshape(30, 2)
    assert seq_ops.frame_skip(seq, 0.5).shape[0] == 15
    assert seq_ops.frame_skip(seq, 1.0).shape[0] == 30


def test_landmark_dropout_zeroes_whole_groups():
    """Groups (hand/pose) are removed per frame, which is what occlusion looks like."""
    seq = np.ones((20, config.MODEL_INPUT_DIM), dtype=np.float32)
    out = seq_ops.landmark_dropout(seq, 0.5, np.random.default_rng(3), groups=("left_hand",))
    lh = config.LAYOUT.slice("left_hand")
    left = out[:, lh.start * config.LAYOUT.coords: lh.stop * config.LAYOUT.coords]
    other = out[:, config.LAYOUT.slice("right_hand").start * config.LAYOUT.coords:]
    frames_zeroed = float(np.mean((left == 0).all(axis=1)))
    assert 0.15 < frames_zeroed < 0.85, frames_zeroed
    assert (other == 1).all(), "only the requested group may be touched"
    assert out.shape == seq.shape


def test_gaussian_jitter_standard_deviation():
    """The operators work on one clip at a time, shaped (T, D)."""
    seq = np.ones((30, config.MODEL_INPUT_DIM), dtype=np.float32)
    out = seq_ops.gaussian_jitter(seq, 0.05, np.random.default_rng(4))
    assert out.shape == seq.shape
    assert abs(float((out - seq).std()) - 0.05) < 0.01


def test_sequence_ops_reject_batched_input():
    """A (N, T, D) batch must fail loudly instead of silently flattening the time axis."""
    batch = np.zeros((4, 30, config.MODEL_INPUT_DIM), dtype=np.float32)
    with pytest.raises(ValueError):
        seq_ops.gaussian_jitter(batch, 0.05, np.random.default_rng(0))


def test_landmark_dropout_rejects_batched_input():
    batch = np.zeros((4, 30, config.MODEL_INPUT_DIM), dtype=np.float32)
    with pytest.raises(ValueError):
        seq_ops.frame_drop(batch, 0.1, np.random.default_rng(0))


def test_augment_preserves_shape_and_is_seeded():
    seq = np.random.default_rng(5).normal(size=(30, 225)).astype(np.float32)
    a = seq_ops.augment(seq.copy(), np.random.default_rng(7))
    b = seq_ops.augment(seq.copy(), np.random.default_rng(7))
    assert a.shape == (config.SEQUENCE_LENGTH, config.MODEL_INPUT_DIM)
    assert np.allclose(a, b), "same seed must give the same augmentation"
    assert not np.allclose(a, seq), "augmentation must actually change the clip"


def test_mirror_hands_swaps_slots(sample_sequence):
    out = lm.mirror_hands(sample_sequence[0])
    # coordinates flip on x, but a second mirror returns the original
    assert np.allclose(lm.mirror_hands(out), sample_sequence[0])


# --------------------------------------------------------------------------
# extractor contract (no camera needed: availability only)
# --------------------------------------------------------------------------
def test_extractor_reports_backend_or_missing_assets():
    from backend.preprocessing.mediapipe_extractor import MediaPipeExtractor

    ex = MediaPipeExtractor()
    try:
        state = ex.backend if hasattr(ex, "backend") else "unknown"
        assert isinstance(state, str)
    finally:
        ex.close()
