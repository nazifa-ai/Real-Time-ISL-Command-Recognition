#!/usr/bin/env python3
"""
PHASE 10 -- FastAPI service.

Endpoints
---------
GET  /health              liveness + whether a trained model could be loaded
GET  /model-info          models on disk, class list, preprocessing + gate settings
GET  /metrics             aggregated audit-log statistics (latency, uncertain rate, ...)
GET  /config              effective runtime settings the UI needs
GET  /classes             the 50 official INCLUDE-50 class names
GET  /dataset/clips       demo clips from the keypoint index (no camera required)
POST /predict             prediction from an uploaded video *or* a JSON payload
POST /feedback            human ACCEPT / CORRECT / REJECT for an event
POST /personalize         create/update a profile, add calibration samples, fine-tune
GET  /profiles            list personalisation profiles
GET  /audit/events        error-review rows (filters: low confidence, rejected, corrected, class)
GET  /audit/pairs         most frequent predicted->corrected pairs

Every prediction written to ``logs/predictions.jsonl`` is a real model output: there
is no fallback that fabricates a label.  If no trained model exists the endpoints
return 503 with the exact training command to run.

Run
---
    uvicorn backend.main:app --host 0.0.0.0 --port 8000
    python backend/main.py            # same, honouring .env via config.py
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
import tempfile
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from backend import personalization as pers  # noqa: E402
from backend.schemas import (  # noqa: E402
    AuditEvent, DatasetClip, ErrorResponse, FeedbackRequest, HealthResponse, MetricsResponse,
    ModelInfoResponse, PersonalizeRequest, PredictRequest, PredictionResponse, ProfileResponse,
)
from backend.inference import audit  # noqa: E402

LOG = logging.getLogger("api")
STARTED_AT = time.time()

# --------------------------------------------------------------------------
# engine cache
# --------------------------------------------------------------------------
_ENGINES: Dict[Tuple[str, Optional[str]], object] = {}
_LAST_ERROR: Dict[str, str] = {}
_LIVE_SESSIONS: Dict[str, Dict[str, object]] = {}
_LIVE_SESSIONS_LOCK = threading.Lock()


def get_engine(model: Optional[str] = None, profile: Optional[str] = None,
               threshold: Optional[float] = None, dataset_clip: bool = False):
    """Return a cached PredictionEngine; raises FileNotFoundError when weights are missing."""
    from backend.inference.predictor import PredictionEngine

    model = model or config.DEFAULT_MODEL
    key = (model, profile if not dataset_clip else None)
    if key not in _ENGINES:
        weights = None
        if profile and not dataset_clip:
            p = pers.profile_dir(profile) / "model.keras"
            weights = p if p.exists() else None
        engine = PredictionEngine(model_name=model,
                                  threshold=(config.CONFIDENCE_THRESHOLD if threshold is None else threshold),
                                  weights_path=weights)
        if profile:
            pers.apply_profile_to_engine(engine, profile)
        _ENGINES[key] = engine
        LOG.info("loaded engine %s (profile=%s, weights=%s)", model, profile, engine.model_path)
    engine = _ENGINES[key]
    if threshold is not None and abs(float(engine.threshold) - float(threshold)) > 1e-9:
        engine.threshold = float(threshold)
    return engine


def clear_engine_cache() -> None:
    _ENGINES.clear()


# --------------------------------------------------------------------------
# input safety
# --------------------------------------------------------------------------
_VIDEO_SIGNATURES = (
    (b"ftyp", 4),        # mp4 / mov / m4v
    (b"RIFF", 0),        # avi
    (b"\x1a\x45\xdf\xa3", 0),  # matroska / webm
    (b"FLV", 0),
)


def _sniff_video(head: bytes) -> bool:
    return any(head[off:off + len(sig)] == sig for sig, off in _VIDEO_SIGNATURES)


def _validate_upload(filename: str, size: int) -> Tuple[str, str]:
    ext = Path(filename or "").suffix.lower()
    if ext not in config.ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                            f"unsupported file type '{ext}'; allowed: {list(config.ALLOWED_VIDEO_EXTENSIONS)}")
    if size > config.MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                            f"file exceeds MAX_UPLOAD_MB={config.MAX_UPLOAD_MB}")
    return ext, filename


def _safe_dataset_clip(rel_path: str) -> str:
    """Only clips that exist in the frozen index may be requested (no arbitrary paths)."""
    from backend.preprocessing.include50 import load_index

    rel_path = (rel_path or "").strip().lstrip("/")
    if ".." in rel_path:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "invalid clip path")
    df = load_index(config.SPLIT_PROTOCOL)
    match = df[df["FilePath"].str.strip() == rel_path]
    if match.empty:
        raise HTTPException(status.HTTP_404_NOT_FOUND,
                            f"'{rel_path}' is not part of split protocol '{config.SPLIT_PROTOCOL}'")
    return str(match.iloc[0]["FilePath"])


def _features_from_upload(data: bytes, filename: str) -> Tuple[np.ndarray, Dict[str, object]]:
    from backend.inference.frames import features_from_video

    ext, _ = _validate_upload(filename, len(data))
    if not _sniff_video(data[:32]):
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                            "file content does not look like a video container")
    tmpdir = Path(tempfile.mkdtemp(prefix="isl_upload_"))
    tmp = tmpdir / f"upload{ext}"
    try:
        tmp.write_bytes(data)
        feats, info = features_from_video(tmp, return_landmarks=True)
        if feats is None:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                                f"no usable landmarks in the upload ({info.get('error')}); "
                                f"need >= {config.MIN_VALID_FRAMES} frames with a detected person")
        return feats, info
    finally:
        # privacy: user videos are not kept unless PERSIST_UPLOADS=1
        if config.PERSIST_UPLOADS:
            keep = Path(config.RAW_DIR) / "uploads"
            keep.mkdir(parents=True, exist_ok=True)
            shutil.copy(tmp, keep / tmp.name)
        shutil.rmtree(tmpdir, ignore_errors=True)


# --------------------------------------------------------------------------
# rate limiting (per client IP, simple sliding window)
# --------------------------------------------------------------------------
_HITS: Dict[str, List[float]] = {}
RATE_LIMIT_PER_MIN = int(os.getenv("API_RATE_LIMIT_PER_MIN", "240"))


def _rate_limit(client: str) -> None:
    now = time.time()
    hits = [t for t in _HITS.get(client, []) if now - t < 60.0]
    if len(hits) >= RATE_LIMIT_PER_MIN:
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS,
                            f"rate limit exceeded ({RATE_LIMIT_PER_MIN}/min)")
    hits.append(now)
    _HITS[client] = hits


# --------------------------------------------------------------------------
# app
# --------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        get_engine()
        LOG.info("default model '%s' loaded at startup", config.DEFAULT_MODEL)
    except Exception as exc:                      # keep the service up, report in /health
        _LAST_ERROR["startup"] = f"{type(exc).__name__}: {exc}"
        LOG.warning("no trained model available at startup: %s", exc)
    yield
    _ENGINES.clear()


app = FastAPI(
    title="Real-Time Indian Sign Language Command Recognition API",
    description=__doc__,
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # demo service; tighten ALLOWED_ORIGINS for deployment
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.middleware("http")
async def _gate(request: Request, call_next):
    client = request.client.host if request.client else "unknown"
    if request.url.path.startswith(("/predict", "/feedback", "/personalize")):
        try:
            _rate_limit(client)
        except HTTPException as exc:
            return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)
    t0 = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Process-Time-Ms"] = f"{(time.perf_counter() - t0) * 1000:.2f}"
    return response


@app.exception_handler(FileNotFoundError)
async def _missing_model(_request: Request, exc: FileNotFoundError):
    return JSONResponse({"detail": str(exc)}, status_code=status.HTTP_503_SERVICE_UNAVAILABLE)


# --------------------------------------------------------------------------
# meta endpoints
# --------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    try:
        engine = get_engine()
        loaded, reason = True, None
        n_classes = len(engine.classes)
        model_name = engine.model_name
    except Exception as exc:
        loaded, reason = False, f"{type(exc).__name__}: {exc}"
        model_name = config.DEFAULT_MODEL
        n_classes = 0
    dataset_ok = False
    try:
        from backend.preprocessing.include50 import keypoint_archive, load_vocabulary

        dataset_ok = keypoint_archive() is not None or bool(load_vocabulary())
    except Exception:
        dataset_ok = False
    try:
        from backend.inference.audio import Speaker

        speech_ok = Speaker().available
    except Exception:
        speech_ok = False
    return HealthResponse(
        status="ok" if loaded else "degraded", model_loaded=loaded, model=model_name,
        num_classes=n_classes, uptime_s=round(time.time() - STARTED_AT, 2),
        dataset_available=dataset_ok, speech_available=speech_ok, reason=reason,
    )


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    from backend.preprocessing.include50 import load_vocabulary

    classes = load_vocabulary()
    pretrained: Dict[str, Dict[str, object]] = {}
    for name in config.AVAILABLE_MODELS:
        d = Path(config.MODEL_PATHS[name])
        info: Dict[str, object] = {"present": (d / "model.keras").exists(), "loadable": False}
        if info["present"]:
            try:
                get_engine(name)
                info["loadable"] = True
            except Exception as exc:
                info["load_error"] = f"{type(exc).__name__}: {exc}"
        hist_path = d / "training_history.json"
        if hist_path.exists():
            import json as _json

            hist = _json.loads(hist_path.read_text())
            info.update({
                "parameters": hist.get("parameters"),
                "epochs_run": hist.get("epochs_run"),
                "best_val_accuracy": hist.get("best_val_accuracy"),
                "training_history": str(hist_path),
            })
        pretrained[name] = info

    evaluation: Dict[str, object] = {}
    summary_path = Path(config.RESULTS_DIR) / "summary_metrics.csv"
    if summary_path.exists():
        import csv as _csv

        with open(summary_path, newline="", encoding="utf-8") as f:
            evaluation["summary_metrics"] = list(_csv.DictReader(f))
    per_class_path = Path(config.RESULTS_DIR) / "per_class_comparison.csv"
    if per_class_path.exists():
        import csv as _class_csv

        with open(per_class_path, newline="", encoding="utf-8") as f:
            evaluation["per_class"] = list(_class_csv.DictReader(f))
    robustness_path = Path(config.RESULTS_DIR) / "robustness_summary.json"
    if robustness_path.exists():
        import json as _json2
        evaluation["robustness"] = _json2.loads(robustness_path.read_text())
    detail_path = Path(config.RESULTS_DIR) / f"evaluation_summary_{config.SPLIT_PROTOCOL}.json"
    if detail_path.exists():
        import json as _json3
        evaluation["detail"] = _json3.loads(detail_path.read_text())
    evaluation["files"] = sorted(p.name for p in Path(config.RESULTS_DIR).glob("*"))

    try:
        from backend.inference.audio import Speaker

        speech = Speaker().status()
    except Exception as exc:
        speech = {"available": False, "reason": str(exc)}

    return ModelInfoResponse(
        default_model=config.DEFAULT_MODEL,
        available_models=[name for name, info in pretrained.items() if info.get("loadable")],
        loaded_models=sorted({k[0] for k in _ENGINES}),
        pretrained=pretrained,
        classes=classes,
        num_classes=len(classes),
        input_shape=[None, config.SEQUENCE_LENGTH, config.MODEL_INPUT_DIM],
        feature_dim=config.FEATURE_DIM,
        sequence_length=config.SEQUENCE_LENGTH,
        confidence_threshold=config.CONFIDENCE_THRESHOLD,
        smoothing={
            "window": config.SMOOTHING_WINDOW, "min_votes": config.SMOOTHING_MIN_VOTES,
            "stable_frames": config.STABLE_FRAMES_REQUIRED,
            "duplicate_suppression": config.DUPLICATE_SUPPRESSION,
            "cooldown_s": config.TTS_COOLDOWN,
        },
        split_protocol=config.SPLIT_PROTOCOL,
        evaluation=evaluation,
        speech=speech,
        personalization_enabled=config.PERSONALIZATION_ENABLED,
    )


@app.get("/metrics", response_model=MetricsResponse)
def metrics() -> MetricsResponse:
    return MetricsResponse(**audit.metrics_summary())


@app.get("/config")
def effective_config() -> Dict[str, object]:
    return {
        "sequence_length": config.SEQUENCE_LENGTH,
        "feature_dim": config.FEATURE_DIM,
        "model_input_dim": config.MODEL_INPUT_DIM,
        "mask_enabled": config.MASK_ENABLED,
        "default_model": config.DEFAULT_MODEL,
        "available_models": list(config.AVAILABLE_MODELS),
        "confidence_threshold": config.CONFIDENCE_THRESHOLD,
        "uncertain_label": config.UNCERTAIN_LABEL,
        "split_protocol": config.SPLIT_PROTOCOL,
        "evaluation": {
            "per_class_comparison": str(Path(config.RESULTS_DIR) / "per_class_comparison.csv"),
            "summary_metrics": str(Path(config.RESULTS_DIR) / "summary_metrics.csv"),
            "robustness_results": str(Path(config.RESULTS_DIR) / "robustness_results.csv"),
        },
        "privacy": {
            "persist_uploads": config.PERSIST_UPLOADS,
            "max_upload_mb": config.MAX_UPLOAD_MB,
            "logging": "append-only JSONL under logs/",
        },
    }


@app.get("/classes")
def classes() -> Dict[str, object]:
    from backend.preprocessing.include50 import load_vocabulary

    v = load_vocabulary()
    return {"num_classes": len(v), "classes": v,
            "source": "datasets/metadata/vocabulary.json (derived from AI4Bharat INCLUDE-50 metadata)"}


@app.get("/dataset/clips", response_model=List[DatasetClip])
def dataset_clips(split: str = Query("test"), label: Optional[str] = None, limit: int = 12) -> List[DatasetClip]:
    from backend.preprocessing.include50 import load_index

    limit = max(1, min(int(limit), 100))
    df = load_index(config.SPLIT_PROTOCOL)
    if split and split != "all":
        df = df[df["split"] == split]
    if label:
        df = df[df["label"].str.upper() == label.upper()]
    rows = df.head(limit)
    return [DatasetClip(path=str(r.FilePath), label=str(r.label), split=str(r.split)) for r in rows.itertuples()]


# --------------------------------------------------------------------------
# prediction
# --------------------------------------------------------------------------
def _top_k(engine, probs: np.ndarray, k: int = 5) -> List[Dict[str, object]]:
    idx = np.argsort(probs)[::-1][:k]
    return [{"class": engine.classes[int(i)], "probability": round(float(probs[int(i)]), 6)}
            for i in idx]


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: Request):
    """Predict from an uploaded video (multipart) or a JSON payload.

    multipart/form-data: ``file=<video>`` plus optional ``model``, ``threshold``,
    ``profile``, ``speak``, ``log`` form fields.
    application/json:    ``PredictRequest`` (features or dataset_clip).
    """
    content_type = (request.headers.get("content-type") or "").split(";")[0].strip()
    source = "api"
    info: Dict[str, object] = {}

    if content_type.startswith("multipart/form-data"):
        form = await request.form()
        upload: Optional[UploadFile] = form.get("file")          # type: ignore[assignment]
        if upload is None or not getattr(upload, "filename", ""):
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "multipart request must include a 'file' video")
        model = (form.get("model") or config.DEFAULT_MODEL)
        profile = form.get("profile") or None
        speak = str(form.get("speak", "false")).lower() in ("1", "true", "yes")
        do_log = str(form.get("log", "true")).lower() in ("1", "true", "yes")
        threshold = form.get("threshold")
        threshold = float(threshold) if threshold not in (None, "") else None
        if model not in config.AVAILABLE_MODELS:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                                f"model must be one of {list(config.AVAILABLE_MODELS)}")
        data = await upload.read()
        feats, info = _features_from_upload(data, upload.filename or "upload.mp4")
        source = "upload"
    else:
        try:
            body = PredictRequest(**await request.json())
        except Exception as exc:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, f"invalid request body: {exc}")
        model = body.model or config.DEFAULT_MODEL
        profile = body.profile
        speak = body.speak
        do_log = body.log
        threshold = body.threshold
        if body.features is not None:
            feats = np.asarray(body.features, dtype=np.float32)
            source = "api_features"
        elif body.dataset_clip:
            from backend.inference.frames import dataset_features

            clip = _safe_dataset_clip(body.dataset_clip)
            feats = dataset_features(clip)
            if feats is None:
                raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                                    f"clip '{clip}' has no usable landmarks")
            info = {"dataset_clip": clip}
            source = "dataset_clip"
        else:
            raise HTTPException(status.HTTP_400_BAD_REQUEST,
                                "provide a video upload, 'features', or 'dataset_clip'")

    engine = get_engine(model, profile, threshold)
    # An API request carries one *complete* clip, not a frame of a stream: the
    # temporal smoother (3-frame vote + stability) cannot be satisfied by a single
    # observation, so the clip-level decision is the gated argmax, while the
    # smoother state is still reported in `stable`.  Streaming entry points
    # (backend/inference/realtime.py, the UI live tab) do use the full smoother.
    engine.clear_window()
    # A video upload is one complete temporal sequence, so it goes through the
    # clip-level confidence gate below. Do not ask the live-frame smoother to
    # decide whether to speak: it intentionally requires several updates.
    result = engine.predict(feats, speak=False)
    probs = engine.probabilities(feats)
    clip_confident = result.raw_confidence >= engine.threshold
    clip_sign = result.raw_sign if clip_confident else config.UNCERTAIN_LABEL
    clip_status = "CONFIDENT" if clip_confident else config.UNCERTAIN_LABEL
    if speak and clip_confident:
        engine.speak_label(clip_sign)

    event = None
    if do_log:
        log_info = {k: v for k, v in info.items() if k != "landmarks"}  # keep the audit
        # log lean -- landmarks are only needed in the immediate API response for the
        # verification overlay, not persisted with every event
        event = audit.log_prediction(
            clip_sign, result.raw_confidence, clip_status, result.latency_ms, result.model,
            source=source, profile=profile, caption=result.caption,
            extra={"raw_sign": result.raw_sign, "raw_confidence": round(result.raw_confidence, 6),
                   "threshold": engine.threshold, "stable": result.stable,
                   "smoothed_sign": result.sign,
                   "input_frames": int(np.asarray(feats).shape[0]), **log_info},
        )

    return PredictionResponse(
        sign=clip_sign, confidence=round(result.raw_confidence, 6), status=clip_status,
        latency_ms=round(result.latency_ms, 3), model=result.model,
        raw_sign=result.raw_sign, raw_confidence=round(result.raw_confidence, 6),
        stable=result.stable, caption=result.caption, threshold=engine.threshold,
        event_id=(event or {}).get("event_id"), source=source,
        top_k=_top_k(engine, probs),
        landmarks=info.get("landmarks"),
    )


@app.post("/live/start")
async def live_start(request: Request) -> Dict[str, object]:
    """Start a privacy-preserving browser camera stream with one persistent tracker."""
    try:
        body = await request.json()
    except Exception:
        body = {}
    model = body.get("model") or config.DEFAULT_MODEL
    if model not in config.AVAILABLE_MODELS:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "unsupported model")
    threshold = body.get("threshold")
    if threshold is not None:
        try:
            threshold = float(threshold)
        except (TypeError, ValueError):
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "threshold must be numeric")
        if not 0.05 <= threshold <= 0.999:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "threshold must be between 0.05 and 0.999")
    profile = body.get("profile") or None
    smoothing_window = body.get("smoothing_window")
    if smoothing_window is not None:
        try:
            smoothing_window = int(smoothing_window)
        except (TypeError, ValueError):
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "smoothing_window must be an integer")
        if smoothing_window not in (1, 3, 5, 7):
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                                "smoothing_window must be one of 1, 3, 5, 7")
    base = get_engine(model, profile, threshold)
    try:
        from backend.inference.frames import FrameBuffer
        from backend.inference.predictor import PredictionEngine, TemporalSmoother
        from backend.preprocessing.mediapipe_extractor import MediaPipeExtractor

        buffer = FrameBuffer(extractor=MediaPipeExtractor())
    except Exception as exc:
        audit.log_failure("mediapipe_init", f"{type(exc).__name__}: {exc}", model=model,
                          source="live-camera")
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE,
                            f"MediaPipe could not start: {type(exc).__name__}: {exc}")
    session_id = uuid.uuid4().hex
    stream_engine = PredictionEngine(model_name=model, model=base.model, classes=base.classes,
                                     threshold=base.threshold, load=False,
                                     # The rolling 30-frame feature window already provides
                                     # temporal context. Require only one stable smoothed
                                     # decision afterward to avoid adding unnecessary delay.
                                     smoother=TemporalSmoother(
                                         window=(smoothing_window or config.SMOOTHING_WINDOW),
                                         min_votes=((smoothing_window // 2 + 1) if smoothing_window else config.SMOOTHING_MIN_VOTES),
                                         stable_frames=1))
    now = time.monotonic()
    with _LIVE_SESSIONS_LOCK:
        expired = [key for key, value in _LIVE_SESSIONS.items()
                   if now - float(value["last_seen"]) > 900.0]
        for key in expired:
            old = _LIVE_SESSIONS.pop(key)
            old["buffer"].close()
        if len(_LIVE_SESSIONS) >= 8:
            buffer.close()
            raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "too many active camera sessions")
        _LIVE_SESSIONS[session_id] = {
            "buffer": buffer, "engine": stream_engine, "model": model, "profile": profile,
            "last_seen": now, "last_uncertain_log": 0.0, "last_frame_at": 0.0,
            "last_event_id": None, "last_event_sign": None, "last_event_status": None,
            "empty_hand_streak": 0,
        }
    return {"session_id": session_id, "sequence_length": config.SEQUENCE_LENGTH,
            "model": model, "status": "ready"}


@app.post("/live/frame")
async def live_frame(request: Request) -> Dict[str, object]:
    """Process exactly one browser JPEG frame through MediaPipe and the rolling model window."""
    request_started = time.perf_counter()
    form = await request.form()
    session_id = str(form.get("session_id") or "")
    with _LIVE_SESSIONS_LOCK:
        session = _LIVE_SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "camera session expired; restart the camera")
    frame_now = time.monotonic()
    if frame_now - float(session["last_frame_at"]) < 0.075:
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "camera frames are arriving too quickly")
    session["last_frame_at"] = frame_now
    upload = form.get("file")
    if upload is None or not hasattr(upload, "read"):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "frame image is required")
    data = await upload.read(config.MAX_UPLOAD_MB * 1024 * 1024 + 1)
    if len(data) > config.MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                            f"file exceeds MAX_UPLOAD_MB={config.MAX_UPLOAD_MB}")
    if not data or len(data) > 1_500_000:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "frame must be between 1 byte and 1.5 MB")
    decode_started = time.perf_counter()
    try:
        import cv2
        image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None or image.shape[0] > 1080 or image.shape[1] > 1920:
            raise ValueError("invalid or oversized image")
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elapsed_ms = int(max(0.0, float(form.get("timestamp_ms") or 0.0)))
        decode_ms = (time.perf_counter() - decode_started) * 1000.0
    except Exception as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, f"invalid camera frame: {exc}")

    buffer = session["buffer"]
    engine = session["engine"]
    try:
        features = buffer.push(rgb, timestamp_ms=elapsed_ms)
        landmarks = buffer.last_landmarks
        from backend.preprocessing.landmarks import pack_frame

        coords, _mask = pack_frame(landmarks)
        landmark_state = {"pose": landmarks.pose is not None,
                          "left_hand": landmarks.left_hand is not None,
                          "right_hand": landmarks.right_hand is not None}
        if landmark_state["left_hand"] or landmark_state["right_hand"]:
            session["empty_hand_streak"] = 0
        else:
            session["empty_hand_streak"] = int(session["empty_hand_streak"]) + 1
            if int(session["empty_hand_streak"]) >= 2:
                # Discard stale hand frames and vote state after sustained tracking
                # loss; do not let a previous sign leak into the next decision.
                buffer.reset()
                engine.smoother.reset()
                features = None
                session["last_event_id"] = None
                session["last_event_sign"] = None
                session["last_event_status"] = None
        response: Dict[str, object] = {
            "ready": features is not None, "fill_ratio": buffer.fill_ratio(),
            "landmark_state": landmark_state, "landmarks": coords.tolist(),
            "camera_fps": round(buffer.stats.fps, 1), "sign": "COLLECTING",
            "status": "PENDING", "confidence": 0.0, "model": session["model"],
            "event_id": None, "should_speak": False,
            "timings_ms": {"decode": round(decode_ms, 3), **buffer.last_stage_timings,
                           "model": 0.0, "hand_gate": 0.0, "smoothing": 0.0},
        }
        if features is not None:
            result = engine.predict(features, speak=False, fps=buffer.stats.fps)
            response.update({"sign": result.sign, "status": result.status,
                             "confidence": round(result.confidence, 6),
                             "raw_sign": result.raw_sign,
                             "raw_confidence": round(result.raw_confidence, 6),
                             "model_latency_ms": round(result.model_latency_ms, 3),
                             "hand_gate_latency_ms": round(result.hand_gate_latency_ms, 3),
                             "smoothing_latency_ms": round(result.smoothing_latency_ms, 3),
                             "top_k": result.top_k,
                             "should_speak": result.should_speak})
            response["timings_ms"].update({
                "model": round(result.model_latency_ms, 3),
                "hand_gate": round(result.hand_gate_latency_ms, 3),
                "smoothing": round(result.smoothing_latency_ms, 3),
            })
            now = time.monotonic()
            if (result.sign, result.status) != (session["last_event_sign"], session["last_event_status"]):
                session["last_event_id"] = None
                session["last_event_sign"] = result.sign
                session["last_event_status"] = result.status
            should_log = result.should_speak or (result.status == config.UNCERTAIN_LABEL
                                                  and now - float(session["last_uncertain_log"]) >= 3.0)
            if should_log:
                event = audit.log_prediction(
                    result.sign, result.confidence, result.status, result.latency_ms,
                    result.model, fps=buffer.stats.fps, source="live-camera",
                    profile=session["profile"], caption=result.caption,
                    extra={"raw_sign": result.raw_sign, "raw_confidence": result.raw_confidence,
                           "threshold": engine.threshold, "sequence_frames": config.SEQUENCE_LENGTH,
                           "stable": result.stable, "session_id": session_id},
                )
                session["last_event_id"] = event["event_id"]
                if result.status == config.UNCERTAIN_LABEL:
                    session["last_uncertain_log"] = now
            response["event_id"] = session["last_event_id"]
        session["last_seen"] = time.monotonic()
        response["processing_ms"] = round((time.perf_counter() - request_started) * 1000.0, 2)
        response["timings_ms"]["server_total"] = response["processing_ms"]
        return response
    except HTTPException:
        raise
    except Exception as exc:
        LOG.exception("live frame failed")
        audit.log_failure("live_frame", f"{type(exc).__name__}: {exc}", model=session.get("model"),
                          source="live-camera", session_id=session_id)
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                            f"camera frame processing failed: {type(exc).__name__}: {exc}")


@app.post("/live/stop")
async def live_stop(request: Request) -> Dict[str, object]:
    try:
        body = await request.json()
    except Exception:
        body = {}
    session_id = str(body.get("session_id") or "")
    with _LIVE_SESSIONS_LOCK:
        session = _LIVE_SESSIONS.pop(session_id, None)
    if session:
        session["buffer"].close()
    return {"status": "stopped"}


@app.post("/live/reset")
async def live_reset(request: Request) -> Dict[str, object]:
    try:
        body = await request.json()
    except Exception:
        body = {}
    session_id = str(body.get("session_id") or "")
    with _LIVE_SESSIONS_LOCK:
        session = _LIVE_SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "camera session expired; restart the camera")
    session["buffer"].reset()
    session["engine"].smoother.reset()
    session["engine"].reset_caption()
    session["empty_hand_streak"] = 0
    session["last_event_id"] = None
    session["last_event_sign"] = None
    session["last_event_status"] = None
    return {"status": "reset", "sequence_length": config.SEQUENCE_LENGTH}


@app.post("/live/model")
async def live_model(request: Request) -> Dict[str, object]:
    body = await request.json()
    session_id = str(body.get("session_id") or "")
    model = str(body.get("model") or "")
    if model not in config.AVAILABLE_MODELS:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                            f"model must be one of {list(config.AVAILABLE_MODELS)}")
    with _LIVE_SESSIONS_LOCK:
        session = _LIVE_SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "camera session expired; restart the camera")
    base = get_engine(model, session.get("profile"), session["engine"].threshold)
    session["engine"].model = base.model
    session["engine"].model_name = model
    session["engine"].classes = base.classes
    session["engine"].threshold = base.threshold
    session["model"] = model
    session["buffer"].reset()
    session["engine"].smoother.reset()
    session["last_event_id"] = None
    session["last_event_sign"] = None
    session["last_event_status"] = None
    return {"status": "switched", "model": model,
            "sequence_length": config.SEQUENCE_LENGTH}


@app.post("/live/settings")
async def live_settings(request: Request) -> Dict[str, object]:
    body = await request.json()
    session_id = str(body.get("session_id") or "")
    threshold = body.get("threshold", config.CONFIDENCE_THRESHOLD)
    window = body.get("smoothing_window", config.SMOOTHING_WINDOW)
    try:
        threshold = float(threshold)
        window = int(window)
    except (TypeError, ValueError):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                            "threshold and smoothing_window must be numeric")
    if not 0.05 <= threshold <= 0.999 or window not in (1, 3, 5, 7):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "settings outside supported range")
    with _LIVE_SESSIONS_LOCK:
        session = _LIVE_SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "camera session expired; restart the camera")
    from backend.inference.predictor import TemporalSmoother

    engine = session["engine"]
    engine.threshold = threshold
    engine.smoother = TemporalSmoother(window=window, min_votes=window // 2 + 1, stable_frames=1)
    session["buffer"].reset()
    session["last_event_id"] = None
    session["last_event_sign"] = None
    session["last_event_status"] = None
    return {"status": "updated", "threshold": threshold, "smoothing_window": window}


# --------------------------------------------------------------------------
# human feedback / error review
# --------------------------------------------------------------------------
@app.post("/feedback")
def feedback(body: FeedbackRequest) -> Dict[str, object]:
    from backend.preprocessing.include50 import load_vocabulary

    classes = load_vocabulary()
    if body.corrected_class and body.corrected_class.strip().upper() not in classes:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                            "corrected_class must be one of the supported vocabulary labels")
    if body.event_id and not any(row.get("event_id") == body.event_id
                                 for row in audit.read_events(limit=1000)):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "prediction event was not found in the audit history")
    try:
        record = audit.log_feedback(body.event_id, body.action, body.predicted_class,
                                    body.corrected_class, body.confidence, body.model,
                                    body.note, body.profile)
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc))
    return {"status": "recorded", **record}


@app.get("/audit/events", response_model=List[AuditEvent])
def audit_events(limit: int = 100, max_confidence: Optional[float] = None,
                 min_confidence: Optional[float] = None, status_filter: Optional[str] = Query(None, alias="status"),
                 predicted_class: Optional[str] = None, action: Optional[str] = None,
                 model: Optional[str] = None, only_corrected: bool = False,
                 only_rejected: bool = False, only_uncertain: bool = False) -> List[AuditEvent]:
    rows = audit.read_events(limit=min(int(limit), 1000), min_confidence=min_confidence,
                             max_confidence=max_confidence, status=status_filter,
                             predicted_class=predicted_class, action=action, model=model,
                             only_corrected=only_corrected, only_rejected=only_rejected,
                             only_uncertain=only_uncertain)
    return [AuditEvent(**{k: r.get(k) for k in AuditEvent.model_fields}) for r in rows]


@app.get("/audit/failures")
def audit_failures(limit: int = 100) -> List[Dict[str, object]]:
    return audit.read_failures(limit=limit)


@app.get("/audit/pairs")
def audit_pairs(min_count: int = 1) -> List[Dict[str, object]]:
    return audit.error_pairs(min_count=min_count)


# --------------------------------------------------------------------------
# personalisation
# --------------------------------------------------------------------------
@app.post("/personalize")
def personalize(body: PersonalizeRequest) -> Dict[str, object]:
    if not config.PERSONALIZATION_ENABLED:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "personalisation disabled by configuration")
    try:
        if body.action == "create":
            profile = pers.create_profile(body.profile, body.base_model, body.vocabulary,
                                          body.threshold, body.preset)
            return {"status": "created", "profile": profile}
        if body.action == "update":
            profile = pers.update_profile(body.profile, body.vocabulary, body.threshold, body.base_model)
            return {"status": "updated", "profile": profile}
        if body.action == "delete":
            return {"status": "deleted" if pers.delete_profile(body.profile) else "not_found"}
        if body.action == "calibrate":
            feats = np.asarray(body.features, dtype=np.float32)
            profile = pers.add_calibration_samples(body.profile, feats, body.labels or [])
            return {"status": "calibrated", "profile": profile}
        if body.action == "fine_tune":
            result = pers.fine_tune(body.profile, epochs=body.epochs)
            clear_engine_cache()          # next request picks up the personalised weights
            return {"status": "fine_tuned", **result}
        if body.action == "reset_fine_tune":
            clear_engine_cache()
            return {"status": "reset", "profile": pers.reset_fine_tune(body.profile)}
    except pers.PersonalizationError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc))
    raise HTTPException(status.HTTP_400_BAD_REQUEST, f"unknown action '{body.action}'")


@app.get("/profiles", response_model=List[ProfileResponse])
def profiles() -> List[ProfileResponse]:
    return [ProfileResponse(**{k: p.get(k, ProfileResponse.model_fields[k].default)
                               for k in ProfileResponse.model_fields})
            for p in pers.list_profiles()]


@app.get("/profiles/{name}", response_model=ProfileResponse)
def profile_detail(name: str) -> ProfileResponse:
    try:
        p = pers.load_profile(name)
    except pers.PersonalizationError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc))
    return ProfileResponse(**{k: p.get(k, ProfileResponse.model_fields[k].default)
                              for k in ProfileResponse.model_fields})


@app.post("/models/switch")
def switch_model(model: str) -> Dict[str, object]:
    if model not in config.AVAILABLE_MODELS:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                            f"model must be one of {list(config.AVAILABLE_MODELS)}")
    get_engine(model)
    return {"status": "loaded", "model": model, "loaded_models": sorted({k[0] for k in _ENGINES})}


@app.get("/static/{path:path}", include_in_schema=False)
def static_files(path: str):
    root = Path(__file__).resolve().parents[1] / "frontend"
    target = (root / path).resolve()
    if root.resolve() not in target.parents or not target.is_file():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "static asset not found")
    return FileResponse(target)


@app.get("/", include_in_schema=False)
def root():
    return FileResponse(Path(__file__).resolve().parents[1] / "frontend" / "index.html")


def main() -> int:
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    uvicorn.run("backend.main:app", host=config.API_HOST, port=config.API_PORT,
                log_level="info", reload=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
