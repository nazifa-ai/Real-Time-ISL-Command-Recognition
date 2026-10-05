"""Model I/O: construction, shapes, parameter counts, saving/loading, determinism."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import config
from training import models as model_zoo


def test_builders_return_expected_output_shape():
    # the builders take the input width and class count explicitly; build_model()
    # supplies the project defaults (225 features, 50 classes)
    assert model_zoo.build_gru(config.MODEL_INPUT_DIM, config.NUM_CLASSES) is not None
    for name in config.AVAILABLE_MODELS:
        model = model_zoo.build_model(name)
        assert tuple(model.input_shape) == (None, None, config.MODEL_INPUT_DIM), name
        assert tuple(model.output_shape) == (None, config.NUM_CLASSES), name


def test_parameter_counts_are_within_budget():
    for name in config.AVAILABLE_MODELS:
        model = model_zoo.build_model(name)
        params = model_zoo.parameter_count(model)
        assert 0 < params < config.MAX_PARAM_BUDGET, name


def test_models_are_comparable_in_size():
    """The proposed model must not win by being an order of magnitude bigger."""
    counts = {n: model_zoo.parameter_count(model_zoo.build_model(n)) for n in config.AVAILABLE_MODELS}
    assert max(counts.values()) / min(counts.values()) < 3.0, counts


def test_gru_mha_contains_attention_and_pooling():
    model = model_zoo.build_model("gru_mha")
    layers = [type(l).__name__ for l in model.layers]
    assert any("MultiHeadAttention" in n for n in layers)
    assert model_zoo.parameter_count(model) > model_zoo.parameter_count(model_zoo.build_model("gru"))


def test_output_is_a_probability_distribution():
    model = model_zoo.build_model("gru")
    x = np.random.default_rng(0).normal(size=(2, config.SEQUENCE_LENGTH, config.MODEL_INPUT_DIM)).astype(np.float32)
    probs = model.predict(x, verbose=0)
    assert probs.shape == (2, config.NUM_CLASSES)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-4)
    assert probs.min() >= 0.0


def test_seeding_makes_initialisation_reproducible():
    from training.train_common import set_seeds

    set_seeds(123)
    a = model_zoo.build_model("gru")
    w1 = [w.copy() for w in a.get_weights()]
    set_seeds(123)
    b = model_zoo.build_model("gru")
    w2 = [w.copy() for w in b.get_weights()]
    assert all(np.allclose(x, y) for x, y in zip(w1, w2))


def test_save_and_load_roundtrip(tmp_path):
    import tensorflow as tf

    model = model_zoo.build_model("lstm")
    path = tmp_path / "tiny.keras"
    model.save(path)
    reloaded = tf.keras.models.load_model(str(path))
    x = np.random.default_rng(1).normal(size=(1, config.SEQUENCE_LENGTH, config.MODEL_INPUT_DIM)).astype(np.float32)
    assert np.allclose(model.predict(x, verbose=0), reloaded.predict(x, verbose=0), atol=1e-6)


def test_models_accept_variable_sequence_length():
    """Input is (None, None, D): live buffers of any length can be scored."""
    model = model_zoo.build_model("gru")
    for t in (10, config.SEQUENCE_LENGTH, 45):
        x = np.zeros((1, t, config.MODEL_INPUT_DIM), dtype=np.float32)
        assert model.predict(x, verbose=0).shape == (1, config.NUM_CLASSES)


def test_unknown_model_name_raises():
    with pytest.raises(KeyError):
        model_zoo.build_model("transformer_xl")


# --------------------------------------------------------------------------
# trained artefacts
# --------------------------------------------------------------------------
def test_trained_models_load_and_predict(trained_model):
    name, model = trained_model
    x = np.zeros((1, config.SEQUENCE_LENGTH, config.MODEL_INPUT_DIM), dtype=np.float32)
    probs = model.predict(x, verbose=0)[0]
    assert probs.shape == (config.NUM_CLASSES,)
    assert abs(float(probs.sum()) - 1.0) < 1e-4


def test_all_trained_models_agree_on_input_shape(all_trained_models):
    for name, model in all_trained_models.items():
        assert model.input_shape[-1] == config.MODEL_INPUT_DIM, name
        assert model.output_shape[-1] == config.NUM_CLASSES, name


def test_training_history_recorded_metrics():
    import json

    for name in config.AVAILABLE_MODELS:
        p = Path(config.MODEL_PATHS[name]) / "training_history.json"
        if not p.exists():
            continue
        h = json.loads(p.read_text())
        assert h["parameters"] > 0
        assert 0.0 <= h["best_val_accuracy"] <= 1.0
        assert len(h["history"]["loss"]) == h["epochs_run"]
        assert h["config"]["seed"] == config.RANDOM_SEED


def test_models_select_one_architecture_each():
    """Different architectures must actually differ in layer composition."""
    def kinds(name):
        return [type(l).__name__ for l in model_zoo.build_model(name).layers]

    assert kinds("lstm") != kinds("gru")
    assert "MultiHeadAttention" in kinds("gru_mha")
    assert "LSTM" in kinds("lstm") and "GRU" in kinds("gru")
