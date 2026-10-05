"""Sequence creation: real clips -> (SEQUENCE_LENGTH, MODEL_INPUT_DIM) tensors."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import config
from backend.preprocessing import include50
from training import preprocess as prep
from training import train_common


def _body_clip(n_frames: int, seed: int = 0) -> tuple:
    rng = np.random.default_rng(seed)
    coords = rng.normal(0, 0.01, size=(n_frames, 75, 3)).astype(np.float32)
    coords[:, 11] = [-0.2, 0.1, 0.0]
    coords[:, 12] = [0.2, 0.1, 0.0]
    coords[:, 23] = [-0.12, -0.25, 0.0]
    coords[:, 24] = [0.12, -0.25, 0.0]
    mask = np.ones((n_frames, 75), dtype=np.float32)
    return coords, mask


def test_clip_to_sequence_shape_and_range():
    coords, mask = _body_clip(60)
    seq = prep.clip_to_sequence(coords, mask, length=config.SEQUENCE_LENGTH)
    assert seq.shape == (config.SEQUENCE_LENGTH, config.MODEL_INPUT_DIM)
    assert np.isfinite(seq).all()
    assert float(np.abs(seq).max()) < 50, "normalisation keeps values in body-scale units"


def test_short_clip_is_upsampled_not_dropped():
    coords, mask = _body_clip(config.MIN_VALID_FRAMES)
    seq = prep.clip_to_sequence(coords, mask, length=config.SEQUENCE_LENGTH)
    assert seq is not None and seq.shape[0] == config.SEQUENCE_LENGTH


def test_too_short_clip_is_rejected():
    coords, mask = _body_clip(config.MIN_VALID_FRAMES - 1)
    assert prep.clip_to_sequence(coords, mask) is None


def test_empty_clip_is_rejected():
    assert prep.clip_to_sequence(np.zeros((0, 75, 3), dtype=np.float32),
                                 np.zeros((0, 75), dtype=np.float32)) is None


def test_sequence_length_is_configurable_not_hard_coded():
    coords, mask = _body_clip(80)
    seq = prep.clip_to_sequence(coords, mask, length=17)
    assert seq.shape == (17, config.MODEL_INPUT_DIM)


def test_build_arrays_from_real_index(index_df):
    X, y, meta, skipped = prep.build_arrays(index_df, "test", augment_copies=0, limit=4)
    assert X.shape[1:] == (config.SEQUENCE_LENGTH, config.MODEL_INPUT_DIM)
    assert X.shape[0] == y.shape[0] == len(meta)
    assert skipped == [], f"real clips should not be skipped: {skipped[:2]}"
    assert set(np.unique(y)) <= set(range(config.NUM_CLASSES))
    assert all("frames_source" in m and m["frames_source"] > 0 for m in meta)


def test_augmented_copies_increase_train_size(index_df):
    X0, _, _, _ = prep.build_arrays(index_df, "train", augment_copies=0, limit=5)
    X1, _, _, _ = prep.build_arrays(index_df, "train", augment_copies=1, limit=5)
    assert X1.shape[0] == 2 * X0.shape[0] == 10


def test_validation_and_test_splits_are_never_augmented(index_df):
    X, _, meta, _ = prep.build_arrays(index_df, "val", augment_copies=2, limit=5)
    assert X.shape[0] == 5
    assert {m["copy"] for m in meta} == {0}


def test_processed_meta_counts_match_index_splits(processed, index_df):
    """val/test are 1:1 with the index; train is (1 + augment_copies) x the index rows."""
    meta = json.loads(str(processed["meta_json"]))
    summary = json.loads(Path("datasets/processed/preprocess_summary_session_disjoint.json").read_text())
    copies = 1 + int(summary.get("augment_copies", 0))
    for split in ("val", "test"):
        n_index = int((index_df["split"] == split).sum())
        assert len(meta[split]) == n_index, f"{split}: {len(meta[split])} clips vs {n_index} index rows"
    n_train_index = int((index_df["split"] == "train").sum())
    assert len(meta["train"]) == n_train_index * copies
    assert {m["copy"] for m in meta["train"]} == set(range(copies))


def test_no_test_clip_is_inside_training_set(processed):
    """Leakage guard: the same source clip must not appear in two splits."""
    meta = json.loads(str(processed["meta_json"]))
    train_paths = {m["filepath"] for m in meta["train"]}
    val_paths = {m["filepath"] for m in meta["val"]}
    test_paths = {m["filepath"] for m in meta["test"]}
    assert not (train_paths & test_paths)
    assert not (train_paths & val_paths)
    assert not (val_paths & test_paths)


def test_load_training_data_matches_npz(processed):
    data = train_common.load_training_data(config.SPLIT_PROTOCOL)
    assert data.X_train.shape == processed["X_train"].shape
    assert data.X_test.shape == processed["X_test"].shape
    assert len(data.classes) == len(processed["classes"])
    assert np.array_equal(data.y_test, processed["y_test"])
    assert data.summary["sequence_length"] == config.SEQUENCE_LENGTH


def test_rebuilt_real_clip_matches_saved_training_tensor(processed):
    """Training and inference preprocessing must agree on the same source clip."""
    meta = json.loads(str(processed["meta_json"]))["test"]
    assert meta, "processed test metadata should identify the source clips"
    coords, mask, _ = include50.read_keypoints_for(meta[0]["filepath"])
    rebuilt = prep.clip_to_sequence(coords, mask)
    assert rebuilt is not None
    assert np.allclose(rebuilt, processed["X_test"][0], rtol=1e-6, atol=1e-6)
