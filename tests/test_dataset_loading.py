"""Dataset loader / metadata parsing / split integrity."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import config
from backend.preprocessing import include50


# --------------------------------------------------------------------------
# metadata
# --------------------------------------------------------------------------
def test_metadata_files_present():
    for key in ("train_50", "test_50"):
        p = Path(config.METADATA_DIR) / config.METADATA_FILES[key]
        assert p.exists(), f"missing official metadata {p}"
        assert p.stat().st_size > 0


def test_vocabulary_length_and_sortedness(vocabulary):
    assert len(vocabulary) == config.NUM_CLASSES == 50
    assert all(v == v.upper() for v in vocabulary), "class labels are stored upper-case"
    assert len(set(vocabulary)) == len(vocabulary), "no duplicate class names"
    # the official INCLUDE-50 list is sorted({Word}) - see include.py in OpenHands
    assert vocabulary == sorted(vocabulary)


def test_vocabulary_matches_official_metadata():
    """Re-derive the class list from the official CSVs and compare with the frozen file."""
    train = include50.load_split_metadata("train", "official")
    test = include50.load_split_metadata("test", "official")
    derived = sorted({str(w) for w in pd.concat([train, test])["label"].unique()})
    assert len(derived) == 50
    assert derived == include50.load_vocabulary()
    # the raw (un-normalised) word stays available for traceability
    assert pd.concat([train, test])["raw_word"].notna().all()


def test_split_metadata_columns():
    df = include50.load_split_metadata("train", "official")
    for col in ("raw_word", "label", "Video", "Category", "FilePath", "split"):
        assert col in df.columns, f"expected column {col} in official metadata"
    # labels are the normalised, upper-case glosses actually used as classes
    assert (df["label"] == df["label"].str.upper()).all()


# --------------------------------------------------------------------------
# index / splits
# --------------------------------------------------------------------------
def test_index_columns_and_size(index_df):
    for col in ("FilePath", "Video", "label", "split", "subset", "Category"):
        assert col in index_df.columns
    assert len(index_df) == 958
    assert set(index_df["split"]) <= {"train", "val", "test"}
    assert index_df["FilePath"].is_unique


def test_index_labels_are_from_vocabulary(index_df, vocabulary):
    assert set(index_df["label"]) <= set(vocabulary)


def test_every_class_in_every_split(index_df):
    for split in ("train", "val", "test"):
        labels = set(index_df[index_df.split == split]["label"])
        assert len(labels) == 50, f"{split} is missing classes: {set(vocabulary) - labels}"


def test_default_protocol_is_session_disjoint():
    assert config.SPLIT_PROTOCOL == "session_disjoint"
    # the alternative protocol is still available for the comparability table
    assert (Path(config.METADATA_DIR) / "index_official.csv").exists()


def test_no_recording_cluster_spans_a_split(index_df):
    """The default protocol must not leak a recording burst across splits."""
    clusters = include50.session_clusters(index_df)
    df = index_df.assign(_cluster=clusters)
    spanning = df.groupby("_cluster")["split"].nunique()
    assert int((spanning > 1).sum()) == 0


def test_official_protocol_leaks_sessions(index_df):
    """Documented property of the official split: sessions *do* span train/test."""
    p = Path(config.METADATA_DIR) / "index_official.csv"
    if not p.exists():
        pytest.skip("official index not built")
    official = pd.read_csv(p)
    clusters = include50.session_clusters(official)
    df = official.assign(_cluster=clusters)
    spanning = df.groupby("_cluster")["split"].nunique()
    assert int((spanning > 1).sum()) > 0


# --------------------------------------------------------------------------
# session helpers
# --------------------------------------------------------------------------
@pytest.mark.parametrize("raw,expected", [
    ("MVI_0031.MOV", "0031"),
    ("MVI_0060_1.MP4", "0060"),
    ("MVI_4155.MOV", "4155"),
])
def test_session_id_regex(raw, expected):
    assert include50.session_id(raw) == expected


def test_session_id_handles_non_numeric_names():
    assert include50.session_id("clip-final.mov") == "clip-final"


def test_session_blocks_group_close_ids():
    df = pd.DataFrame({"FilePath": [f"Greetings/48. Hello/MVI_{i}.MOV" for i in (100, 105, 130)]})
    blocks = df["FilePath"].map(lambda p: include50.session_block(p, block=10))
    assert blocks.nunique() == 2


# --------------------------------------------------------------------------
# reader
# --------------------------------------------------------------------------
def test_read_keypoints_for_real_clip():
    df = include50.load_index(config.SPLIT_PROTOCOL)
    row = df[df.split == "test"].iloc[0]
    coords, mask, info = include50.read_keypoints_for(row["FilePath"])
    assert coords.ndim == 3 and coords.shape[1:] == (75, 3)
    assert mask.shape == (coords.shape[0], 75)
    assert coords.shape[0] >= config.MIN_VALID_FRAMES
    assert np.isfinite(coords).all()
    assert "vid_shape" in info and "x_axis_scale" in info
