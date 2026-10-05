"""FastAPI surface: health, model info, metrics, validation, predict, feedback, personalise."""

from __future__ import annotations

import io

import numpy as np
import pytest

import config


# --------------------------------------------------------------------------
# health / metadata
# --------------------------------------------------------------------------
def test_health(api_client):
    r = api_client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] in ("ok", "degraded")
    assert "model_loaded" in body and "uptime_s" in body
    assert "num_classes" in body, "health reports the class count under an unambiguous name"
    if body["model_loaded"]:
        assert body["num_classes"] == config.NUM_CLASSES


def test_live_session_starts_with_dynamic_time_axis_model(api_client):
    """The saved models declare a dynamic time axis but live sends 30 frames."""
    response = api_client.post("/live/start", json={
        "model": config.DEFAULT_MODEL, "threshold": 0.70, "smoothing_window": 3,
    })
    assert response.status_code == 200, response.text
    session_id = response.json()["session_id"]
    assert response.json()["sequence_length"] == config.SEQUENCE_LENGTH
    stopped = api_client.post("/live/stop", json={"session_id": session_id})
    assert stopped.status_code == 200


def test_root_serves_live_camera_dashboard(api_client):
    page = api_client.get("/")
    script = api_client.get("/static/app.js")
    assert page.status_code == 200 and 'id="startBtn"' in page.text
    assert script.status_code == 200 and "startLiveLoop" in script.text


def test_model_info(api_client):
    r = api_client.get("/model-info")
    assert r.status_code == 200
    body = r.json()
    assert body["num_classes"] == config.NUM_CLASSES
    assert len(body["classes"]) == config.NUM_CLASSES
    assert body["sequence_length"] == config.SEQUENCE_LENGTH
    assert body["split_protocol"] == config.SPLIT_PROTOCOL
    assert body["default_model"] == config.DEFAULT_MODEL
    assert "window" in body["smoothing"] and "speech" in body


def test_classes_endpoint_matches_vocabulary(api_client):
    body = api_client.get("/classes").json()
    assert body["num_classes"] == 50 and len(body["classes"]) == 50
    assert "INCLUDE" in body["source"].upper()


def test_config_endpoint_exposes_no_secrets(api_client):
    body = api_client.get("/config").json()
    text = str(body).lower()
    for secret in ("api_key", "password", "token", "secret"):
        assert secret not in text
    assert body["privacy"]["persist_uploads"] in (True, False)


def test_metrics_endpoint_shape(api_client):
    body = api_client.get("/metrics").json()
    for key in ("predictions_total", "feedback_total", "status_counts", "latency_ms",
                "uncertain_rate", "human_actions", "corrections_total"):
        assert key in body


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------
def test_predict_rejects_empty_body(api_client):
    r = api_client.post("/predict", json={})
    assert r.status_code == 400


def test_predict_rejects_wrong_feature_width(api_client):
    r = api_client.post("/predict", json={"features": [[0.0] * 10] * 30})
    assert r.status_code == 422
    assert str(config.MODEL_INPUT_DIM) in r.text


def test_predict_rejects_too_few_frames(api_client):
    r = api_client.post("/predict", json={"features": [[0.0] * config.MODEL_INPUT_DIM] * 2})
    assert r.status_code == 422


def test_predict_rejects_both_sources(api_client):
    payload = {"features": [[0.0] * config.MODEL_INPUT_DIM] * 30, "dataset_clip": "x/y.MOV"}
    assert api_client.post("/predict", json=payload).status_code == 422


def test_predict_rejects_path_traversal(api_client):
    r = api_client.post("/predict", json={"dataset_clip": "../../etc/passwd"})
    assert r.status_code == 400


def test_predict_rejects_unknown_clip(api_client):
    r = api_client.post("/predict", json={"dataset_clip": "Nope/1. Fake/MVI_0001.MOV"})
    assert r.status_code == 404


def test_predict_rejects_bad_model_name(api_client):
    r = api_client.post("/predict", json={"features": [[0.0] * config.MODEL_INPUT_DIM] * 30,
                                          "model": "resnet"})
    assert r.status_code == 422


def test_upload_rejects_disallowed_extension(api_client):
    r = api_client.post("/predict", files={"file": ("evil.sh", b"#!/bin/sh\n", "text/plain")})
    assert r.status_code == 415


def test_upload_rejects_fake_video_content(api_client):
    r = api_client.post("/predict", files={"file": ("fake.mp4", b"not a video at all", "video/mp4")})
    assert r.status_code == 415


def test_upload_rejects_oversized_file(api_client, monkeypatch):
    monkeypatch.setattr(config, "MAX_UPLOAD_MB", 0)
    data = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 64
    r = api_client.post("/predict", files={"file": ("big.mp4", data, "video/mp4")})
    assert r.status_code == 413


def test_feedback_validation(api_client):
    assert api_client.post("/feedback", json={"action": "MAYBE", "predicted_class": "HELLO"}).status_code == 422
    assert api_client.post("/feedback", json={"action": "CORRECT", "predicted_class": "HELLO"}).status_code == 422
    ok = api_client.post("/feedback", json={"action": "ACCEPT", "predicted_class": "HELLO",
                                            "confidence": 0.9, "model": "gru_mha"})
    assert ok.status_code == 200 and ok.json()["status"] == "recorded"


def test_personalize_validation(api_client):
    bad_vocab = api_client.post("/personalize", json={
        "profile": "t1", "action": "create", "vocabulary": ["NOT_A_REAL_SIGN"]})
    assert bad_vocab.status_code == 422
    bad_name = api_client.post("/personalize", json={"profile": "../etc", "action": "create"})
    assert bad_name.status_code == 422
    bad_preset = api_client.post("/personalize", json={"profile": "t1", "action": "create",
                                                       "preset": "made_up"})
    assert bad_preset.status_code == 422
    calibrate_without_data = api_client.post("/personalize", json={"profile": "t1",
                                                                   "action": "calibrate"})
    assert calibrate_without_data.status_code == 422


# --------------------------------------------------------------------------
# real predictions (skipped when no trained model is on disk)
# --------------------------------------------------------------------------
@pytest.fixture()
def require_model(api_client):
    info = api_client.get("/health").json()
    if not info["model_loaded"]:
        pytest.skip(f"no trained model: {info.get('reason')}")
    return info


def test_predict_from_dataset_clip(api_client, require_model):
    classes = api_client.get("/classes").json()["classes"]
    clip = api_client.get("/dataset/clips", params={"split": "test", "limit": 1}).json()[0]
    r = api_client.post("/predict", json={"dataset_clip": clip["path"], "log": False})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["sign"] in classes or body["sign"] == config.UNCERTAIN_LABEL
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["status"] in ("CONFIDENT", "UNCERTAIN", "PENDING")
    assert body["latency_ms"] > 0
    assert body["model"] == config.DEFAULT_MODEL
    assert len(body["top_k"]) == 5
    probs = [t["probability"] for t in body["top_k"]]
    assert probs == sorted(probs, reverse=True)


def test_predict_logs_an_event(api_client, require_model, temp_logs):
    from backend.inference import audit

    clip = api_client.get("/dataset/clips", params={"split": "test", "limit": 1}).json()[0]
    body = api_client.post("/predict", json={"dataset_clip": clip["path"], "log": True}).json()
    assert body["event_id"]
    rows = audit.read_events(limit=5)
    assert rows and rows[0]["event_id"] == body["event_id"]
    assert rows[0]["source"] == "dataset_clip"


def test_feedback_links_to_prediction_event(api_client, require_model, temp_logs):
    from backend.inference import audit

    clips = api_client.get("/dataset/clips", params={"split": "test", "limit": 2}).json()
    pred = api_client.post("/predict", json={"dataset_clip": clips[0]["path"]}).json()
    api_client.post("/feedback", json={"action": "CORRECT", "predicted_class": pred["sign"],
                                       "corrected_class": clips[1]["label"],
                                       "event_id": pred["event_id"],
                                       "confidence": pred["confidence"], "model": pred["model"]})
    rows = audit.read_events(only_corrected=True)
    assert rows and rows[0]["corrected_class"] == clips[1]["label"]
    pairs = api_client.get("/audit/pairs").json()
    assert pairs and pairs[0]["predicted"] == pred["sign"]


def test_predict_threshold_override_is_reported(api_client, require_model):
    clip = api_client.get("/dataset/clips", params={"split": "test", "limit": 1}).json()[0]
    body = api_client.post("/predict", json={"dataset_clip": clip["path"], "threshold": 0.3,
                                             "log": False}).json()
    assert body["threshold"] == 0.3


def test_live_camera_frame_opens_mediapipe_session(api_client, require_model):
    import cv2

    started = api_client.post("/live/start", json={"model": config.DEFAULT_MODEL})
    assert started.status_code == 200, started.text
    session_id = started.json()["session_id"]
    ok, encoded = cv2.imencode(".jpg", np.zeros((360, 640, 3), dtype=np.uint8))
    assert ok
    try:
        response = api_client.post("/live/frame", data={
            "session_id": session_id, "timestamp_ms": "100",
        }, files={"file": ("camera-frame.jpg", encoded.tobytes(), "image/jpeg")})
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["ready"] is False
        assert len(body["landmarks"]) == config.NUM_LANDMARKS
        assert body["processing_ms"] > 0
        assert body["landmark_state"] == {"pose": False, "left_hand": False, "right_hand": False}
    finally:
        assert api_client.post("/live/stop", json={"session_id": session_id}).status_code == 200


def test_zero_threshold_is_preserved_by_engine_cache(api_client, require_model):
    from backend import main as api

    api.clear_engine_cache()
    engine = api.get_engine(threshold=0.0)
    assert engine.threshold == 0.0


def test_audit_events_filters_via_api(api_client, require_model, temp_logs):
    clip = api_client.get("/dataset/clips", params={"split": "test", "limit": 1}).json()[0]
    api_client.post("/predict", json={"dataset_clip": clip["path"], "log": True})
    assert len(api_client.get("/audit/events", params={"limit": 10}).json()) >= 1
    assert api_client.get("/audit/events", params={"only_rejected": True}).json() == []
