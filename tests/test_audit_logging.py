"""JSONL audit logging, feedback joins, filters and metrics aggregation."""

from __future__ import annotations

import json

import pytest

from backend.inference import audit


def _event(**kw):
    base = dict(predicted_class="HELLO", confidence=0.91, status="CONFIDENT",
                latency_ms=12.5, model="gru_mha", fps=24.0, source="test")
    base.update(kw)
    return audit.log_prediction(**base)


def test_prediction_record_has_all_required_fields(temp_logs):
    rec = _event()
    for field in ("event_id", "timestamp", "model", "predicted_class", "confidence", "status",
                  "latency_ms", "fps", "human_action", "corrected_class"):
        assert field in rec, field
    assert rec["human_action"] is None and rec["corrected_class"] is None
    assert rec["latency_ms"] == 12.5 and rec["confidence"] == 0.91


def test_jsonl_is_append_only_and_parseable(temp_logs):
    for _ in range(3):
        _event()
    lines = audit.predictions_path().read_text().strip().split("\n")
    assert len(lines) == 3
    assert all(json.loads(line)["predicted_class"] == "HELLO" for line in lines)


def test_feedback_requires_a_valid_action(temp_logs):
    ev = _event()
    with pytest.raises(ValueError):
        audit.log_feedback(ev["event_id"], "MAYBE", "HELLO")
    with pytest.raises(ValueError):
        audit.log_feedback(ev["event_id"], "CORRECT", "HELLO")          # missing corrected_class
    rec = audit.log_feedback(ev["event_id"], "correct", "HELLO", "BANK")
    assert rec["human_action"] == "CORRECT" and rec["corrected_class"] == "BANK"


def test_events_join_feedback(temp_logs):
    ev = _event()
    audit.log_feedback(ev["event_id"], "CORRECT", "HELLO", "BANK", confidence=0.91, model="gru_mha")
    rows = audit.read_events()
    assert len(rows) == 1
    assert rows[0]["human_action"] == "CORRECT" and rows[0]["corrected_class"] == "BANK"
    assert rows[0]["wrong"] is True


def test_reject_marks_event_wrong(temp_logs):
    ev = _event()
    audit.log_feedback(ev["event_id"], "REJECT", "HELLO")
    rows = audit.read_events(only_rejected=True)
    assert len(rows) == 1 and rows[0]["wrong"] is True


def test_error_review_filters(temp_logs):
    good = _event(confidence=0.95)
    low = _event(confidence=0.31, status="UNCERTAIN", predicted_class="BANK")
    audit.log_feedback(good["event_id"], "ACCEPT", "HELLO")
    audit.log_feedback(low["event_id"], "CORRECT", "BANK", "BIRD")

    assert len(audit.read_events(max_confidence=0.5)) == 1
    assert len(audit.read_events(min_confidence=0.9)) == 1
    assert len(audit.read_events(only_uncertain=True)) == 1
    assert len(audit.read_events(only_corrected=True)) == 1
    assert len(audit.read_events(predicted_class="BANK")) == 1
    assert len(audit.read_events(action="ACCEPT")) == 1
    assert len(audit.read_events(model="gru_mha")) == 2
    assert audit.read_events(predicted_class="NOT_A_CLASS") == []


def test_events_are_newest_first(temp_logs):
    for i in range(5):
        _event(predicted_class=f"CLASS{i}")
    rows = audit.read_events()
    assert [r["predicted_class"] for r in rows][:2] == ["CLASS4", "CLASS3"]


def test_metrics_summary_math(temp_logs):
    for conf, lat in ((0.9, 10.0), (0.8, 20.0), (0.3, 30.0)):
        _event(confidence=conf, latency_ms=lat,
               status="CONFIDENT" if conf > 0.7 else "UNCERTAIN")
    m = audit.metrics_summary()
    assert m["predictions_total"] == 3
    assert abs(m["confidence"]["mean"] - (0.9 + 0.8 + 0.3) / 3) < 1e-9
    assert abs(m["latency_ms"]["mean"] - 20.0) < 1e-9
    assert m["status_counts"]["UNCERTAIN"] == 1
    assert abs(m["uncertain_rate"] - 1 / 3) < 1e-9
    assert abs(m["throughput_fps"] - 1000 / 20.0) < 1e-6


def test_metrics_human_agreement(temp_logs):
    evs = [_event() for _ in range(4)]
    audit.log_feedback(evs[0]["event_id"], "ACCEPT", "HELLO")
    audit.log_feedback(evs[1]["event_id"], "ACCEPT", "HELLO")
    audit.log_feedback(evs[2]["event_id"], "REJECT", "HELLO")
    audit.log_feedback(evs[3]["event_id"], "CORRECT", "HELLO", "BANK")
    m = audit.metrics_summary()
    assert m["human_agreement_rate"] == 0.5
    assert m["human_actions"] == {"ACCEPT": 2, "CORRECT": 1, "REJECT": 1}
    assert m["corrections_total"] == 1


def test_error_pairs_ranking(temp_logs):
    for _ in range(2):
        ev = _event(predicted_class="HELLO")
        audit.log_feedback(ev["event_id"], "CORRECT", "HELLO", "BANK")
    ev = _event(predicted_class="BANK")
    audit.log_feedback(ev["event_id"], "CORRECT", "BANK", "PLACE")
    pairs = audit.error_pairs()
    assert pairs[0] == {"predicted": "HELLO", "corrected": "BANK", "count": 2}


def test_clear_logs_removes_files(temp_logs):
    _event()
    assert audit.predictions_path().exists()
    audit.clear_logs("all")
    assert not audit.predictions_path().exists()
    assert audit.read_events() == []


def test_reading_missing_log_returns_empty(tmp_path, monkeypatch):
    import config

    monkeypatch.setitem(config.LOG_PATHS, "predictions", tmp_path / "nope.jsonl")
    monkeypatch.setitem(config.LOG_PATHS, "feedback", tmp_path / "nope2.jsonl")
    assert audit.read_events() == []
    assert audit.metrics_summary()["predictions_total"] == 0
