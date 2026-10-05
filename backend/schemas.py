"""
Pydantic request/response models for the FastAPI service (Pydantic v2).

Everything the API accepts is validated here: label strings, feature tensor shape,
upload metadata, feedback actions and personalisation payloads.  The validators are
deliberately strict because these objects are the service's trust boundary.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, Sequence

import config
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# --------------------------------------------------------------------------
# requests
# --------------------------------------------------------------------------
class PredictRequest(BaseModel):
    """Prediction input.

    Exactly one of ``features`` / ``dataset_clip`` should be supplied; when the
    endpoint is called with a multipart upload the file replaces both.
    """

    model: Optional[Literal["lstm", "gru", "gru_mha"]] = None
    features: Optional[List[List[float]]] = Field(
        default=None, description=f"T x {config.MODEL_INPUT_DIM} normalised landmark features")
    dataset_clip: Optional[str] = Field(
        default=None, description="INCLUDE keypoint clip path, e.g. 'Animals/4. Bird/MVI_2987.MOV'")
    threshold: Optional[float] = Field(default=None, ge=0.05, le=0.999)
    speak: bool = False
    log: bool = True
    profile: Optional[str] = Field(default=None, max_length=32)

    @field_validator("features")
    @classmethod
    def _check_features(cls, v: Optional[List[List[float]]]) -> Optional[List[List[float]]]:
        if v is None:
            return v
        if not v:
            raise ValueError("features must not be empty")
        t = len(v)
        if t < config.MIN_VALID_FRAMES:
            raise ValueError(f"need at least {config.MIN_VALID_FRAMES} frames, got {t}")
        if t > config.MAX_SEQUENCE_LENGTH:
            raise ValueError(f"at most {config.MAX_SEQUENCE_LENGTH} frames, got {t}")
        for row in v:
            if len(row) != config.MODEL_INPUT_DIM:
                raise ValueError(
                    f"each frame must have {config.MODEL_INPUT_DIM} values, got {len(row)}")
        return v

    @model_validator(mode="after")
    def _one_source(self) -> "PredictRequest":
        if self.features is not None and self.dataset_clip:
            raise ValueError("provide either features or dataset_clip, not both")
        return self


class FeedbackRequest(BaseModel):
    action: Literal["ACCEPT", "CORRECT", "REJECT"]
    predicted_class: str = Field(min_length=1, max_length=64)
    corrected_class: Optional[str] = Field(default=None, max_length=64)
    event_id: Optional[str] = Field(default=None, max_length=32)
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    model: Optional[str] = Field(default=None, max_length=32)
    note: Optional[str] = Field(default=None, max_length=500)
    profile: Optional[str] = Field(default=None, max_length=32)

    @model_validator(mode="after")
    def _correct_needs_class(self) -> "FeedbackRequest":
        if self.action == "CORRECT" and not (self.corrected_class or "").strip():
            raise ValueError("CORRECT requires corrected_class")
        return self


class PersonalizeRequest(BaseModel):
    profile: str = Field(min_length=1, max_length=32)
    action: Literal["create", "update", "calibrate", "fine_tune", "reset_fine_tune", "delete"]
    base_model: Optional[Literal["lstm", "gru", "gru_mha"]] = None
    vocabulary: Optional[List[str]] = None
    threshold: Optional[float] = Field(default=None, ge=0.05, le=0.999)
    preset: Optional[str] = Field(default=None, max_length=32)
    features: Optional[List[List[float]]] = None
    labels: Optional[List[str]] = None
    epochs: Optional[int] = Field(default=None, ge=1, le=50)

    @model_validator(mode="after")
    def _calibrate_needs_data(self) -> "PersonalizeRequest":
        if self.action == "calibrate":
            if self.features is None or not self.labels:
                raise ValueError("calibrate requires features and labels")
        return self


# --------------------------------------------------------------------------
# responses
# --------------------------------------------------------------------------
class TopKItem(BaseModel):
    """One entry of the top-k distribution; exposed as {"class": ..., "probability": ...}."""

    model_config = ConfigDict(populate_by_name=True)

    class_name: str = Field(validation_alias="class", serialization_alias="class")
    probability: float


class PredictionResponse(BaseModel):
    sign: str
    confidence: float
    status: str
    latency_ms: float
    model: str
    raw_sign: Optional[str] = None
    raw_confidence: Optional[float] = None
    stable: bool = False
    caption: str = ""
    threshold: float = config.CONFIDENCE_THRESHOLD
    event_id: Optional[str] = None
    source: Optional[str] = None
    top_k: Optional[List[TopKItem]] = None
    landmarks: Optional[List[List[List[float]]]] = None  # (timesteps, landmarks, xyz) -- the
    # exact, resampled coordinates behind the feature sequence the model consumed for this
    # prediction. Present only for video-upload requests where extraction happened server-side.


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    model: str
    num_classes: int
    uptime_s: float
    dataset_available: bool
    speech_available: bool
    reason: Optional[str] = None


class ModelInfoResponse(BaseModel):
    default_model: str
    available_models: List[str]
    loaded_models: List[str]
    pretrained: Dict[str, Dict[str, object]]
    classes: List[str]
    num_classes: int
    input_shape: List[object]
    feature_dim: int
    sequence_length: int
    confidence_threshold: float
    smoothing: Dict[str, object]
    split_protocol: str
    evaluation: Dict[str, object]
    speech: Dict[str, object]
    personalization_enabled: bool


class MetricsResponse(BaseModel):
    predictions_total: int
    feedback_total: int
    status_counts: Dict[str, int]
    uncertain_rate: float
    confidence: Dict[str, float]
    latency_ms: Dict[str, float]
    throughput_fps: float
    human_actions: Dict[str, int]
    corrections_total: int
    human_agreement_rate: Optional[float]
    top_predicted_classes: Dict[str, int]
    per_model: Dict[str, int]


class AuditEvent(BaseModel):
    event_id: Optional[str] = None
    timestamp: Optional[str] = None
    model: Optional[str] = None
    predicted_class: Optional[str] = None
    confidence: Optional[float] = None
    status: Optional[str] = None
    latency_ms: Optional[float] = None
    fps: Optional[float] = None
    human_action: Optional[str] = None
    corrected_class: Optional[str] = None
    source: Optional[str] = None
    profile: Optional[str] = None
    note: Optional[str] = None
    stable: Optional[bool] = None
    session_id: Optional[str] = None
    raw_sign: Optional[str] = None
    raw_confidence: Optional[float] = None
    threshold: Optional[float] = None
    sequence_frames: Optional[int] = None


class ProfileResponse(BaseModel):
    name: str
    base_model: str
    vocabulary: List[str] = []
    threshold: float
    calibration_samples: int = 0
    per_class_samples: Dict[str, int] = {}
    fine_tuned: bool = False
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    notes: Optional[str] = None
    fine_tune_history: Optional[Sequence[Dict[str, object]]] = None


class DatasetClip(BaseModel):
    path: str
    label: str
    split: str


class ErrorResponse(BaseModel):
    detail: str
