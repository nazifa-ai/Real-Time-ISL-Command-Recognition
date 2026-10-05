"""Class mapping: one vocabulary, one index space, shared by data, models and API."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import config
from backend.preprocessing import include50


def test_class_mapping_is_vocabulary_order(vocabulary):
    mapping = include50.class_mapping()
    assert list(mapping.keys()) == vocabulary
    assert list(mapping.values()) == list(range(len(vocabulary)))


def test_dataset_metadata_class_names_are_uppercase_glosses(vocabulary):
    for name in vocabulary:
        assert name == name.upper().strip()
        assert len(name) <= 32


def test_vocabulary_file_records_provenance():
    payload = json.loads((Path(config.METADATA_DIR) / "vocabulary.json").read_text())
    assert payload["num_classes"] == 50
    assert "INCLUDE" in payload["dataset"].upper()
    assert payload["citation"] and payload["doi"], "the dataset must be cited, not just used"
    assert len(payload["classes"]) == 50
    assert all(c["videos"] > 0 for c in payload["classes"])


def test_processed_classes_are_vocabulary(processed, vocabulary):
    assert [str(c) for c in processed["classes"]] == vocabulary


def test_label_mapping_file_matches_vocabulary():
    p = Path(config.PROCESSED_DIR) / "label_mapping.json"
    if not p.exists():
        import pytest

        pytest.skip("label_mapping.json not written yet")
    payload = json.loads(p.read_text())
    assert payload["classes"] == include50.load_vocabulary()
    assert payload["num_classes"] == 50


def test_model_class_mapping_matches_vocabulary(vocabulary):
    for name in config.AVAILABLE_MODELS:
        p = Path(config.MODEL_PATHS[name]) / "class_mapping.json"
        if not p.exists():
            continue
        payload = json.loads(p.read_text())
        assert payload["classes"] == vocabulary, f"{name}: class order differs from the vocabulary"
        assert payload["num_classes"] == len(vocabulary)


def test_model_output_width_matches_num_classes(all_trained_models):
    for name, model in all_trained_models.items():
        assert model.output_shape[-1] == config.NUM_CLASSES, name
        assert model.input_shape[-1] == config.MODEL_INPUT_DIM, name


def test_preprocessing_config_is_recorded_per_model():
    for name in config.AVAILABLE_MODELS:
        p = Path(config.MODEL_PATHS[name]) / "preprocessing_config.json"
        if not p.exists():
            continue
        cfg = json.loads(p.read_text())
        assert cfg["sequence_length"] == config.SEQUENCE_LENGTH
        assert cfg["model_input_dim"] == config.MODEL_INPUT_DIM
        assert cfg["split_protocol"] == config.SPLIT_PROTOCOL


def test_label_indices_are_dense(processed):
    assert set(np.unique(processed["y_train"]).tolist()) == set(range(config.NUM_CLASSES))
    for key in ("y_val", "y_test"):
        present = set(np.unique(processed[key]).tolist())
        assert present <= set(range(config.NUM_CLASSES))
        assert len(present) == config.NUM_CLASSES, f"{key} should be class-stratified"
