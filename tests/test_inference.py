"""Confidence gating, temporal smoothing, stability, duplicate suppression, captioning."""

from __future__ import annotations

import numpy as np
import pytest

import config
from backend.inference.predictor import PredictionEngine, TemporalSmoother


def _active_features() -> np.ndarray:
    """(T, MODEL_INPUT_DIM) with real hand-landmark motion, so the new IDLE gate
    (see PredictionEngine._hand_activity) does not intercept these engine tests --
    an all-zero window is IDLE by design (no hands present), which is correct
    production behaviour but not what these tests are exercising."""
    rng = np.random.default_rng(0)
    x = np.zeros((config.SEQUENCE_LENGTH, config.MODEL_INPUT_DIM), dtype=np.float32)
    hand_cols = slice(config.LAYOUT.slice("left_hand").start * config.LAYOUT.coords,
                       config.LAYOUT.slice("right_hand").stop * config.LAYOUT.coords)
    width = hand_cols.stop - hand_cols.start
    x[:, hand_cols] = rng.normal(loc=0.3, scale=0.05, size=(config.SEQUENCE_LENGTH, width)).astype(np.float32)
    return x


# --------------------------------------------------------------------------
# smoothing unit behaviour
# --------------------------------------------------------------------------
def test_single_vote_is_not_enough():
    """One observation cannot satisfy min_votes, so the state stays UNCERTAIN at first."""
    s = TemporalSmoother(window=3, min_votes=2, stable_frames=1)
    assert s.update("HELLO", 0.9)["label"] == config.UNCERTAIN_LABEL
    assert s.update("HELLO", 0.95)["label"] == "HELLO"


def test_majority_vote_beats_outlier():
    s = TemporalSmoother(window=3, min_votes=2, stable_frames=1)
    s.update("BANK", 0.9)
    s.update("BANK", 0.9)
    out = s.update("HELLO", 0.9)
    assert out["label"] == "BANK"
    assert out["votes"] == 2


def test_window_is_bounded():
    s = TemporalSmoother(window=3)
    for lbl in "ABCABC":
        s.update(lbl, 0.9)
    assert len(s.window) == 3


def test_stable_frames_required_before_emit():
    """A label must clear the vote threshold *and* persist for N updates.

    With window=3/min_votes=2 the label is not even a candidate on the first update,
    so a 3-update stability requirement is first satisfied on the 4th update.
    """
    s = TemporalSmoother(window=3, min_votes=2, stable_frames=3)
    outs = [s.update("HELLO", 0.9) for _ in range(4)]
    assert [o["label"] for o in outs] == [config.UNCERTAIN_LABEL, "HELLO", "HELLO", "HELLO"]
    assert [o["stable"] for o in outs] == [False, False, False, True]
    assert [o["should_emit"] for o in outs] == [False, False, False, True]


def test_duplicate_suppression_prevents_repeat():
    s = TemporalSmoother(window=3, min_votes=2, stable_frames=2)
    emitted = [s.update("HELLO", 0.9)["should_emit"] for _ in range(6)]
    assert emitted.count(True) == 1, emitted      # exactly one emission for a held sign
    assert emitted.index(True) == 2               # first stable update
    assert emitted[-1] is False                   # held sign is not shouted again


def test_cooldown_allows_repeat_after_expiry(monkeypatch):
    monkeypatch.setattr(config, "TTS_COOLDOWN", 0.0)
    s = TemporalSmoother(window=1, min_votes=1, stable_frames=1)
    assert s.update("HELLO", 0.9, now=1000.0)["should_emit"] is True
    assert s.update("HELLO", 0.9, now=1001.0)["should_emit"] is True


def test_uncertain_never_emits():
    """The engine gates first, then feeds labels to the smoother: UNCERTAIN must stay UNCERTAIN."""
    s = TemporalSmoother(window=3, min_votes=2, stable_frames=1)
    for _ in range(5):
        out = s.update(config.UNCERTAIN_LABEL, 0.2)
    assert out["label"] == config.UNCERTAIN_LABEL
    assert out["should_emit"] is False
    assert out["stable"] is False


def test_reset_clears_state():
    s = TemporalSmoother()
    s.update("HELLO", 0.9)
    s.reset()
    assert len(s.window) == 0 and s.update("HELLO", 0.9)["streak"] == 1


# --------------------------------------------------------------------------
# engine behaviour with a real model
# --------------------------------------------------------------------------
def test_engine_threshold_gates_prediction(trained_model):
    name, model = trained_model
    eng = PredictionEngine(model_name=name, model=model, threshold=0.999)
    x = _active_features()
    r = eng.predict(x, speak=False)
    assert r.status in (config.UNCERTAIN_LABEL, "PENDING")
    assert r.confidence < 0.999
    assert r.sign == config.UNCERTAIN_LABEL or r.raw_confidence < 0.999


def test_fast_batch_path_matches_keras_predict(trained_model):
    name, model = trained_model
    eng = PredictionEngine(model_name=name, model=model, threshold=0.0)
    x = _active_features()
    expected = model.predict(x[None, ...], verbose=0)[0]
    actual = eng.probabilities(x)
    assert np.allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_engine_below_zero_threshold_always_confident(trained_model):
    name, model = trained_model
    eng = PredictionEngine(model_name=name, model=model, threshold=0.0)
    x = _active_features()
    r = eng.predict(x, speak=False)
    assert r.raw_sign in eng.classes
    assert 0.0 <= r.raw_confidence <= 1.0


def test_engine_caption_and_stability_over_repeated_frames(trained_model):
    name, model = trained_model
    eng = PredictionEngine(model_name=name, model=model, threshold=0.0)
    x = _active_features()
    statuses = [eng.predict(x, speak=False).status for _ in range(6)]
    assert "CONFIDENT" in statuses, statuses
    assert len(eng.caption) == 1, "a held sign must be captioned once, not repeatedly"


def test_engine_reports_latency_and_throughput(trained_model):
    name, model = trained_model
    eng = PredictionEngine(model_name=name, model=model)
    x = _active_features()
    r = eng.predict(x, speak=False)
    stats = eng.stats()
    assert r.latency_ms > 0
    assert stats["mean_latency_ms"] > 0
    assert stats["throughput_fps"] > 0


def test_engine_speech_disabled_never_calls_tts(trained_model, monkeypatch):
    name, model = trained_model
    eng = PredictionEngine(model_name=name, model=model, threshold=0.0)
    called = []

    class Boom:
        available = True

        def say(self, *a, **k):
            called.append(a)
            raise RuntimeError("tts exploded")

    eng._speaker = Boom()
    x = _active_features()
    for _ in range(5):
        eng.predict(x, speak=True)          # must never raise
    assert called, "speech was attempted"
    assert eng.caption, "captioning must continue even when TTS fails"


def test_vocab_filter_restricts_decoding(trained_model):
    name, model = trained_model
    eng = PredictionEngine(model_name=name, model=model, threshold=0.0)
    kept = eng.set_vocab_filter(["HELLO", "BANK"])
    assert set(kept) == {"HELLO", "BANK"}
    assert kept == [c for c in eng.classes if c in {"HELLO", "BANK"}], "kept in vocabulary order"
    x = _active_features()
    probs = eng.probabilities(x)
    assert probs.shape == (config.NUM_CLASSES,)
    assert abs(float(probs.sum()) - 1.0) < 1e-6
    allowed = {eng.classes.index(c) for c in kept}
    assert sum(probs[i] for i in range(len(eng.classes)) if i not in allowed) < 1e-9
    assert eng.set_vocab_filter(None) == []


def test_pop_last_removes_caption_token(trained_model):
    name, model = trained_model
    eng = PredictionEngine(model_name=name, model=model, threshold=0.0)
    eng.caption.extend(["HELLO", "BANK"])
    assert eng.pop_last() == "BANK"
    assert eng.caption == ["HELLO"]
