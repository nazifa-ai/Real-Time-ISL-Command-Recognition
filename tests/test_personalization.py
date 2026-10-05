"""Personalisation rules: vocabulary restriction, threshold clamp, calibration, fine-tune."""

from __future__ import annotations

import numpy as np
import pytest

import config
from backend import personalization as pers


@pytest.fixture(autouse=True)
def isolated_profiles(tmp_path, monkeypatch):
    """Never write profiles into the repository during tests."""
    monkeypatch.setattr(pers, "PROFILE_ROOT", tmp_path / "personalized")
    return tmp_path / "personalized"


# --------------------------------------------------------------------------
# profile lifecycle
# --------------------------------------------------------------------------
def test_create_profile_defaults():
    p = pers.create_profile("alice", base_model="gru")
    assert p["base_model"] == "gru"
    assert p["threshold"] == config.CONFIDENCE_THRESHOLD
    assert p["vocabulary"] == []
    assert p["calibration_samples"] == 0
    assert "not proven" in p["notes"]


@pytest.mark.parametrize("name", ["../etc", "with space", "", "x" * 40, "semi;colon"])
def test_profile_name_validation(name):
    with pytest.raises(pers.PersonalizationError):
        pers.create_profile(name)


def test_unknown_base_model_rejected():
    with pytest.raises(pers.PersonalizationError):
        pers.create_profile("bob", base_model="gpt")


def test_vocabulary_must_come_from_include50():
    with pytest.raises(pers.PersonalizationError):
        pers.create_profile("bob", vocabulary=["TOTALLY_MADE_UP"])
    p = pers.create_profile("bob", vocabulary=["hello", "bank"])       # case-insensitive in
    assert set(p["vocabulary"]) == {"HELLO", "BANK"}


def test_preset_vocabulary():
    p = pers.create_profile("carol", preset="greetings")
    assert p["vocabulary"] == config.PERSONALIZATION_DEFAULT_VOCAB_PRESETS["greetings"]
    assert p["vocabulary"], "a preset must yield a non-empty subset"


def test_every_preset_entry_is_an_official_class():
    """A preset must never invent a sign that the dataset does not contain."""
    from backend.preprocessing.include50 import load_vocabulary

    official = set(load_vocabulary())
    for name, entries in config.PERSONALIZATION_DEFAULT_VOCAB_PRESETS.items():
        assert set(entries) <= official, f"preset '{name}' has unknown entries: {set(entries) - official}"


def test_threshold_is_clamped_to_a_small_band():
    p = pers.create_profile("dave", threshold=0.99)
    assert p["threshold"] == pytest.approx(
        config.CONFIDENCE_THRESHOLD + config.PERSONALIZATION_MAX_THRESHOLD_DELTA)
    assert p["threshold_requested"] == 0.99
    q = pers.create_profile("dave2", threshold=0.0)
    assert q["threshold"] == pytest.approx(
        config.CONFIDENCE_THRESHOLD - config.PERSONALIZATION_MAX_THRESHOLD_DELTA)


def test_update_and_delete_profile():
    pers.create_profile("erin")
    updated = pers.update_profile("erin", vocabulary=["BANK"], threshold=0.75)
    assert updated["vocabulary"] == ["BANK"] and updated["threshold"] == pytest.approx(0.75)
    assert pers.delete_profile("erin") is True
    assert pers.delete_profile("erin") is False
    with pytest.raises(pers.PersonalizationError):
        pers.load_profile("erin")


def test_list_profiles_returns_created_ones():
    pers.create_profile("frank")
    pers.create_profile("gina")
    assert {p["name"] for p in pers.list_profiles()} >= {"frank", "gina"}


# --------------------------------------------------------------------------
# calibration samples
# --------------------------------------------------------------------------
def _samples(n: int, classes=("HELLO", "BANK")) -> tuple:
    rng = np.random.default_rng(0)
    X = rng.normal(0, 0.1, size=(n, config.SEQUENCE_LENGTH, config.MODEL_INPUT_DIM)).astype(np.float32)
    labels = [classes[i % len(classes)] for i in range(n)]
    return X, labels


def test_add_calibration_samples_counts_per_class():
    pers.create_profile("hina", vocabulary=["HELLO", "BANK", "BANK"])
    X, labels = _samples(6)
    p = pers.add_calibration_samples("hina", X, labels)
    assert p["calibration_samples"] == 6
    assert p["per_class_samples"] == {"HELLO": 3, "BANK": 3}
    Xs, ys = pers.load_calibration("hina")
    assert Xs.shape == (6, config.SEQUENCE_LENGTH, config.MODEL_INPUT_DIM)
    assert ys.shape == (6,)


def test_calibration_appends_instead_of_overwriting():
    pers.create_profile("iris", vocabulary=["HELLO"])
    X, labels = _samples(2, ("HELLO",))
    pers.add_calibration_samples("iris", X, labels)
    p = pers.add_calibration_samples("iris", X, labels)
    assert p["calibration_samples"] == 4


def test_calibration_rejects_wrong_shape():
    pers.create_profile("jack", vocabulary=["HELLO"])
    with pytest.raises(pers.PersonalizationError):
        pers.add_calibration_samples("jack", np.zeros((3, 30, 100), dtype=np.float32), ["HELLO"] * 3)


def test_calibration_rejects_unknown_label():
    pers.create_profile("kim", vocabulary=["HELLO"])
    X, _ = _samples(1, ("HELLO",))
    with pytest.raises(pers.PersonalizationError):
        pers.add_calibration_samples("kim", X, ["NOT_A_SIGN"])


def test_calibration_rejects_label_outside_profile_vocabulary():
    pers.create_profile("lee", vocabulary=["HELLO"])
    X, _ = _samples(1, ("BANK",))
    with pytest.raises(pers.PersonalizationError):
        pers.add_calibration_samples("lee", X, ["BANK"])


def test_calibration_label_count_must_match():
    pers.create_profile("mo", vocabulary=["HELLO"])
    X, _ = _samples(2, ("HELLO",))
    with pytest.raises(pers.PersonalizationError):
        pers.add_calibration_samples("mo", X, ["HELLO"])


# --------------------------------------------------------------------------
# gating and fine-tuning
# --------------------------------------------------------------------------
def test_can_fine_tune_requires_minimum_samples():
    pers.create_profile("nina", vocabulary=["HELLO", "BANK"])
    ok, why = pers.can_fine_tune("nina")
    assert ok is False and "at least" in why
    X, labels = _samples(config.PERSONALIZATION_MIN_SAMPLES_FOR_FINETUNE)
    pers.add_calibration_samples("nina", X, labels)
    ok, why = pers.can_fine_tune("nina")
    assert ok is True, why


def test_compile_vocabulary_keeps_index_space():
    pers.create_profile("omar", vocabulary=["BANK", "HELLO"])
    classes = pers.compile_vocabulary(pers.load_profile("omar"))
    assert set(classes) == {"BANK", "HELLO"}
    # index space stays the global one, so model logits can be masked safely
    from backend.preprocessing.include50 import load_vocabulary

    assert classes == [c for c in load_vocabulary() if c in {"BANK", "HELLO"}]


def test_apply_profile_to_engine(trained_model):
    from backend.inference.predictor import PredictionEngine

    name, model = trained_model
    pers.create_profile("pia", vocabulary=["HELLO", "BANK"], threshold=0.8)
    engine = PredictionEngine(model_name=name, model=model)
    applied = pers.apply_profile_to_engine(engine, "pia")
    assert applied["threshold"] == pytest.approx(0.8)
    assert set(engine.vocab_filter) == {"HELLO", "BANK"}
    x = np.zeros((config.SEQUENCE_LENGTH, config.MODEL_INPUT_DIM), dtype=np.float32)
    probs = engine.probabilities(x)
    outside = [i for i, c in enumerate(engine.classes) if c not in ("HELLO", "BANK")]
    assert float(probs[outside].sum()) < 1e-9


def test_fine_tune_end_to_end_and_reset(trained_model):
    """Optional path: needs a real trained base model and enough calibration samples."""
    name, _ = trained_model
    base = pers.create_profile("quinn", base_model=name, vocabulary=["HELLO", "BANK"])
    assert base["fine_tuned"] is False
    X, labels = _samples(config.PERSONALIZATION_MIN_SAMPLES_FOR_FINETUNE, ("HELLO", "BANK"))
    pers.add_calibration_samples("quinn", X, labels)
    result = pers.fine_tune("quinn", epochs=1)
    assert result["samples_used"] == config.PERSONALIZATION_MIN_SAMPLES_FOR_FINETUNE
    assert result["train_samples"] > 0
    assert 0.0 <= result["train_accuracy_on_calibration"] <= 1.0
    assert "NOT evidence" in result["caveat"]
    assert (pers.profile_dir("quinn") / "model.keras").exists()
    assert pers.load_profile("quinn")["fine_tuned"] is True

    reset = pers.reset_fine_tune("quinn")
    assert reset["fine_tuned"] is False
    assert not (pers.profile_dir("quinn") / "model.keras").exists()


def test_fine_tune_refuses_without_samples():
    pers.create_profile("rita", vocabulary=["HELLO", "BANK"])
    with pytest.raises(pers.PersonalizationError):
        pers.fine_tune("rita", epochs=1)
