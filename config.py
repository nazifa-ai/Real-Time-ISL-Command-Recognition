"""
Central configuration for the ISL Recognition project.

Every module reads its constants from here (or from the environment) so that no
value is hard-coded in the middle of the codebase.

Environment overrides
---------------------
Any setting can be overridden with an environment variable prefixed ``ISL_``,
e.g. ``ISL_SEQUENCE_LENGTH=40 ISL_CONFIDENCE_THRESHOLD=0.8 python run.py ...``.
Values are parsed back into the declared type.  A ``.env`` file in the project
root is loaded automatically when ``python-dotenv`` is installed.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Tuple

# --------------------------------------------------------------------------
# Optional .env loading (soft dependency)
# --------------------------------------------------------------------------
try:  # pragma: no cover - trivial
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent / ".env")
except Exception:  # pragma: no cover
    pass


def _env(key: str, default: Any) -> Any:
    """Read ``ISL_<key>`` from the environment, coercing to ``type(default)``."""
    raw = os.getenv(f"ISL_{key}")
    if raw is None or raw == "":
        return default
    if isinstance(default, bool):
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(default, int):
        return int(raw)
    if isinstance(default, float):
        return float(raw)
    if isinstance(default, (tuple, list)):
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        if isinstance(default, tuple):
            return tuple(int(p) if p.isdigit() else p for p in parts)
        return parts
    return raw


# --------------------------------------------------------------------------
# Project identity / dataset
# --------------------------------------------------------------------------
PROJECT_NAME = "Real-Time Indian Sign Language Command Recognition with Temporal Deep Learning"
PROJECT_SLUG = "isl-recognition"

#: Primary dataset.  The vocabulary is *derived from the dataset metadata*, never
#: hard-coded: see ``backend/preprocessing/include50.py``.
DATASET_NAME = _env("DATASET_NAME", "AI4Bharat INCLUDE-50")
DATASET_CITATION = (
    "Sridhar, A., Ganesan, R. G., Kumar, P., Khapra, M. (2020). INCLUDE: A Large Scale "
    "Dataset for Indian Sign Language Recognition. ACM Multimedia 2020."
)
DATASET_DOI = "10.1145/3394171.3413528"

ROOT_DIR = Path(__file__).resolve().parent

#: Raw videos are large (57 GB for full INCLUDE).  By default they are expected
#: inside the repo, but ``ISL_DATASET_PATH`` lets you keep them anywhere.
DATASET_PATH = Path(_env("DATASET_PATH", str(ROOT_DIR / "datasets" / "raw" / "INCLUDE")))
#: Official AI4Bharat pre-extracted MediaPipe keypoints (0.6 GB, Zenodo 6674324).
KEYPOINTS_PATH = Path(_env("KEYPOINTS_PATH", str(DATASET_PATH / "keypoints" / "INCLUDE")))
#: Additional locations searched when KEYPOINTS_PATH does not exist (the shipped
#: archive is a single zip; extracted copies are looked up here too).
KEYPOINTS_FALLBACKS: Tuple[Path, ...] = tuple(
    Path(p) for p in _env("KEYPOINTS_FALLBACKS", (
        str(DATASET_PATH / "keypoints"),
        str(DATASET_PATH / "Pose_Signs"),
        "/data/include/keypoints/Pose_Signs",
        "/data/include/keypoints",
    ))
)


DATASETS_DIR = ROOT_DIR / "datasets"
RAW_DIR = DATASETS_DIR / "raw"
PROCESSED_DIR = Path(_env("PROCESSED_DIR", str(DATASETS_DIR / "processed")))
METADATA_DIR = DATASETS_DIR / "metadata"

MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"
LOGS_DIR = ROOT_DIR / "logs"
FEEDBACK_DIR = Path(_env("FEEDBACK_DIR", str(LOGS_DIR / "feedback")))
DOCS_DIR = ROOT_DIR / "docs"
EVALUATION_DIR = ROOT_DIR / "evaluation"

MODEL_PATHS: Dict[str, Path] = {
    "lstm": MODELS_DIR / "lstm",
    "gru": MODELS_DIR / "gru",
    "gru_mha": MODELS_DIR / "gru_mha",
}
LOG_PATHS: Dict[str, Path] = {
    "predictions": LOGS_DIR / "predictions.jsonl",
    "feedback": FEEDBACK_DIR / "feedback.jsonl",
    "audit": LOGS_DIR / "audit.jsonl",
}

#: Number of output classes. The authoritative value is the frozen vocabulary
#: (datasets/metadata/vocabulary.json), which is itself derived from the official
#: INCLUDE-50 metadata; the literal 50 is only a fallback for a fresh checkout.
def _num_classes() -> int:
    try:
        payload = json.loads((METADATA_DIR / "vocabulary.json").read_text())
        return len(payload["classes"])
    except Exception:
        return 50


NUM_CLASSES: int = _env("NUM_CLASSES", _num_classes())

METADATA_FILES: Dict[str, str] = {
    "train_50": "train_include50.csv",
    "test_50": "test_include50.csv",
    "train_all": "train_include.csv",
    "test_all": "test_include.csv",
    "gloss_norm": "normalized_glosses.csv",
    "hf_train": "include_train.parquet",
    "hf_val": "include_val.parquet",
    "hf_test": "include_test.parquet",
}


# --------------------------------------------------------------------------
# Landmark representation
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class LandmarkLayout:
    """Canonical landmark layout used by *every* part of the system.

    The dataset (and the live camera) may deliver fewer points than the canonical
    75; the remaining slots are written as zeros and flagged in the mask channel.
    """

    #: (name -> (slot_offset, count)) in canonical order
    groups: Tuple[Tuple[str, int, int], ...] = (
        ("pose", 0, 33),        # MediaPipe BlazePose, 33 points
        ("left_hand", 33, 21),  # MediaPipe Hands (left)
        ("right_hand", 54, 21), # MediaPipe Hands (right)
    )
    coords: int = 3  # x, y, z

    @property
    def num_landmarks(self) -> int:
        return sum(count for _, _, count in self.groups)

    @property
    def feature_dim(self) -> int:
        return self.num_landmarks * self.coords

    def slice(self, group: str) -> slice:
        for name, offset, count in self.groups:
            if name == group:
                return slice(offset, offset + count)
        raise KeyError(f"unknown landmark group: {group}")

    def describe(self) -> str:
        return " + ".join(f"{n}({c})" for n, _, c in self.groups) + f" = {self.num_landmarks} landmarks"


LAYOUT = LandmarkLayout()

#: 33 pose + 21 left hand + 21 right hand = 75 landmarks -> 225 features/frame
NUM_LANDMARKS: int = LAYOUT.num_landmarks
COORDS_PER_LANDMARK: int = LAYOUT.coords
FEATURE_DIM: int = LAYOUT.feature_dim

#: Optional per-landmark presence mask appended to the coordinates.
#: ``MASK_ENABLED=True`` -> model input is FEATURE_DIM + NUM_LANDMARKS per frame.
MASK_ENABLED: bool = _env("MASK_ENABLED", False)
MODEL_INPUT_DIM: int = FEATURE_DIM + (NUM_LANDMARKS if MASK_ENABLED else 0)

#: MediaPipe BlazePose landmark indices used for normalisation / geometry.
POSE_NOSE = 0
POSE_LEFT_SHOULDER = 11
POSE_RIGHT_SHOULDER = 12
POSE_LEFT_ELBOW = 13
POSE_RIGHT_ELBOW = 14
POSE_LEFT_WRIST = 15
POSE_RIGHT_WRIST = 16
POSE_LEFT_HIP = 23
POSE_RIGHT_HIP = 24

#: MediaPipe visibility below this value counts as "landmark missing".
VISIBILITY_THRESHOLD: float = _env("VISIBILITY_THRESHOLD", 0.5)

#: Confidence threshold applied to the *released AI4Bharat* keypoints.  Those
#: files store hard 0/1 presence flags per landmark (verified: min 0, max 1), so
#: any value in (0, 1] gives the same mask; it is configurable for symmetry with
#: live MediaPipe, whose visibility scores are continuous.
KEYPOINT_CONFIDENCE_THRESHOLD: float = _env("KEYPOINT_CONFIDENCE_THRESHOLD", 0.5)

# --------------------------------------------------------------------------
# Geometry calibration for the official keypoint release
# --------------------------------------------------------------------------
#: Where landmark data comes from.
#:   "ai4bharat" - official pre-extracted keypoints (Zenodo 6674324), 2-D + z
#:   "mediapipe" - extract from raw INCLUDE videos with this project's extractor
KEYPOINT_SOURCE: str = _env("KEYPOINT_SOURCE", "ai4bharat")

#: !! MEASURED, NOT ASSUMED !!  The released AI4Bharat keypoints are stored in a
#: frame that was resized 1920x1080 -> 1080x1920 (an OpenCV width/height swap), so
#: the archived x axis is compressed relative to y.  Multiplying x by
#: (W/H) ** 2 = 3.1605 restores isotropic geometry that matches live MediaPipe.
#: Empirically confirmed on the dataset by two independent anatomical priors
#: (face nose->eye-line / inter-ocular = 3.00; shoulder-width / torso-length =
#: 3.26) and by the constant spread of the measurement across videos
#: (median 0.2018, IQR 0.1973-0.2047, n=197).  See docs/dataset_audit.md.
X_AXIS_SCALE: float = _env("X_AXIS_SCALE", 3.1605)
APPLY_KEYPOINT_AXIS_CORRECTION: bool = _env("APPLY_KEYPOINT_AXIS_CORRECTION", True)

#: Isotropy references measured with MediaPipe (Tasks API, holistic landmarker)
#: on full-body reference photographs, in true pixels.  Used only to *validate*
#: X_AXIS_SCALE; never used as model input.  See training/calibrate_geometry.py.
MEDIAPIPE_REFERENCE: Dict[str, float] = {
    "shoulder_torso_ratio": 0.6566,  # |sh_L - sh_R| / |mid_shoulder - mid_hip|
    "face_vertical_over_interocular": 0.5238,  # nose->eye-line midpoint / inter-ocular
}


# --------------------------------------------------------------------------
# Temporal sequence
# --------------------------------------------------------------------------
SEQUENCE_LENGTH: int = _env("SEQUENCE_LENGTH", 30)          # frames per sequence
SEQUENCE_STRIDE: int = _env("SEQUENCE_STRIDE", 30)          # window stride (train)
SEQ_RESAMPLE_METHOD: str = _env("SEQ_RESAMPLE_METHOD", "linear")
#: hardest cap accepted from an API client (longer inputs are rejected, not cropped)
MAX_SEQUENCE_LENGTH: int = _env("MAX_SEQUENCE_LENGTH", 600)
SEQ_PAD_MODE: str = _env("SEQ_PAD_MODE", "edge")            # 'edge' | 'zero'
MIN_VALID_FRAMES: int = _env("MIN_VALID_FRAMES", 5)         # reject shorter clips
MAX_FRAMES_READ: int = _env("MAX_FRAMES_READ", 300)         # cap video decoding


# --------------------------------------------------------------------------
# Splits / reproducibility
# --------------------------------------------------------------------------
RANDOM_SEED: int = _env("RANDOM_SEED", 42)
#: fraction of the official *train* set carved out for validation (seeded, stratified)
VAL_FRACTION: float = _env("VAL_FRACTION", 0.10)
#: 'session_disjoint' (default) -> class-stratified, recording-cluster-disjoint split
#:                      over the full INCLUDE-50 pool; the project's primary evaluation
#:                      because the official split leaks across recording sessions
#: 'official'          -> official AI4Bharat train/test files (+ seeded validation carve-out);
#:                      kept for comparability with published numbers, reported as leaked
#: 'hf'                -> HuggingFace INCLUDE train/val/test files
SPLIT_PROTOCOL: str = _env("SPLIT_PROTOCOL", "session_disjoint")
#: target test fraction for the session-disjoint split (train gets the remainder)
TEST_FRACTION: float = _env("TEST_FRACTION", 0.15)


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
BATCH_SIZE: int = _env("BATCH_SIZE", 32)
EPOCHS: int = _env("EPOCHS", 80)
LEARNING_RATE: float = _env("LEARNING_RATE", 1e-3)
EARLY_STOPPING_PATIENCE: int = _env("EARLY_STOPPING_PATIENCE", 12)
REDUCE_LR_PATIENCE: int = _env("REDUCE_LR_PATIENCE", 5)
REDUCE_LR_FACTOR: float = _env("REDUCE_LR_FACTOR", 0.5)
MIN_LEARNING_RATE: float = _env("MIN_LEARNING_RATE", 1e-5)
DROPOUT: float = _env("DROPOUT", 0.3)
LSTM_UNITS: Tuple[int, int] = _env("LSTM_UNITS", (128, 64))
GRU_UNITS: Tuple[int, int] = _env("GRU_UNITS", (128, 64))
ATTENTION_HEADS: int = _env("ATTENTION_HEADS", 4)
ATTENTION_KEY_DIM: int = _env("ATTENTION_KEY_DIM", 64)
DENSE_UNITS: int = _env("DENSE_UNITS", 128)
LABEL_SMOOTHING: float = _env("LABEL_SMOOTHING", 0.05)
MAX_PARAM_BUDGET: int = _env("MAX_PARAM_BUDGET", 5_000_000)  # sanity guard for tests

#: Augmentation policy - identical for all three models (fair comparison).
AUGMENTATION: Dict[str, Any] = {
    "enabled": _env("AUGMENT_ENABLED", True),
    "jitter_sigma": _env("AUG_JITTER_SIGMA", 0.01),      # gaussian noise on normalised coords
    "frame_drop_prob": _env("AUG_FRAME_DROP_PROB", 0.10), # randomly repeat neighbours
    "landmark_dropout_prob": _env("AUG_LM_DROPOUT", 0.05),
    "temporal_crop_frac": _env("AUG_TEMPORAL_CROP", 0.10),
    "mirror_prob": _env("AUG_MIRROR_PROB", 0.0),          # off: handedness is semantic in ISL
    "augment_copies": _env("AUG_COPIES", 1),
}


# --------------------------------------------------------------------------
# Inference / UX behaviour
# --------------------------------------------------------------------------
#: Initial confidence gate.  NOT claimed to be optimal - it is tuned and reported
#: by ``training/tune_threshold.py`` / documented in ``docs/evaluation.md``.
CONFIDENCE_THRESHOLD: float = _env("CONFIDENCE_THRESHOLD", 0.70)
SMOOTHING_WINDOW: int = _env("SMOOTHING_WINDOW", 3)      # 3-frame majority vote
SMOOTHING_MIN_VOTES: int = _env("SMOOTHING_MIN_VOTES", 2)
STABLE_FRAMES_REQUIRED: int = _env("STABLE_FRAMES_REQUIRED", 3)
TTS_COOLDOWN: float = _env("TTS_COOLDOWN", 3.0)          # seconds between identical utterances
DUPLICATE_SUPPRESSION: bool = _env("DUPLICATE_SUPPRESSION", True)
UNCERTAIN_LABEL: str = "UNCERTAIN"
IDLE_LABEL: str = "IDLE"

#: Fraction of frames in the window that must show *any* hand landmark
#: (non-zero magnitude) before the engine will even attempt classification.
#: Below this, the window is reported as IDLE instead of forcing the model
#: to pick one of its trained classes for empty/resting input.
IDLE_MIN_HAND_PRESENCE_RATIO: float = _env("IDLE_MIN_HAND_PRESENCE_RATIO", 0.30)
#: Mean absolute frame-to-frame change in the hand landmarks, below which the
#: hands are considered "present but static" (resting, not actively signing)
#: and the window is reported as IDLE rather than classified.
IDLE_MIN_HAND_MOTION: float = _env("IDLE_MIN_HAND_MOTION", 0.01)

DEFAULT_MODEL: str = _env("DEFAULT_MODEL", "gru_mha")
AVAILABLE_MODELS: Tuple[str, ...] = ("lstm", "gru", "gru_mha")

TTS_ENABLED_DEFAULT: bool = _env("TTS_ENABLED", True)
TTS_ENGINE: str = _env("TTS_ENGINE", "auto")  # auto | gtts | pyttsx3 | none
TTS_LANGUAGE: str = _env("TTS_LANGUAGE", "en")
TTS_RATE: int = _env("TTS_RATE", 165)

CAMERA_INDEX: int = _env("CAMERA_INDEX", 0)
#: how many extra indices to try when the preferred camera is unavailable
CAMERA_PROBE_MAX_INDEX: int = _env("CAMERA_PROBE_MAX_INDEX", 3)
FRAME_WIDTH: int = _env("FRAME_WIDTH", 640)
FRAME_HEIGHT: int = _env("FRAME_HEIGHT", 480)
TARGET_FPS: int = _env("TARGET_FPS", 30)


# --------------------------------------------------------------------------
# API / frontend
# --------------------------------------------------------------------------
API_HOST: str = _env("API_HOST", "0.0.0.0")
API_PORT: int = _env("API_PORT", 8000)
API_BASE_URL: str = _env("API_BASE_URL", "http://localhost:8000")
MAX_UPLOAD_MB: int = _env("MAX_UPLOAD_MB", 200)
ALLOWED_VIDEO_EXTENSIONS: Tuple[str, ...] = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v")
ALLOWED_IMAGE_EXTENSIONS: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
PERSIST_UPLOADS: bool = _env("PERSIST_UPLOADS", False)   # privacy: do not keep user videos
REQUEST_TIMEOUT_S: float = _env("REQUEST_TIMEOUT_S", 120.0)


# --------------------------------------------------------------------------
# Personalisation (see docs/system_card.md - kept deliberately conservative)
# --------------------------------------------------------------------------
PERSONALIZATION_ENABLED: bool = _env("PERSONALIZATION_ENABLED", True)
PERSONALIZATION_MIN_SAMPLES_FOR_FINETUNE: int = _env("PERSONALIZATION_MIN_SAMPLES", 20)
PERSONALIZATION_FINETUNE_EPOCHS: int = _env("PERSONALIZATION_FINETUNE_EPOCHS", 5)
PERSONALIZATION_FINETUNE_LR: float = _env("PERSONALIZATION_FINETUNE_LR", 1e-4)
PERSONALIZATION_MAX_THRESHOLD_DELTA: float = _env("PERSONALIZATION_MAX_THRESHOLD_DELTA", 0.2)
PERSONALIZATION_DEFAULT_VOCAB_PRESETS: Dict[str, list] = {
    # Presets only *filter* the official INCLUDE-50 vocabulary; they never invent a
    # sign.  Every entry below is validated against datasets/metadata/vocabulary.json
    # by tests/test_personalization.py.
    "all": [],
    "greetings": ["HELLO", "GOOD_MORNING", "THANK_YOU", "GOOD"],
    "people": ["ME", "YOU_PLURAL", "BOY", "GIRL", "BROTHER", "FATHER", "TEACHER",
               "PRIEST"],
    "objects": ["CAR", "CELL_PHONE", "HAT", "HOUSE", "PEN", "SHOE", "T-SHIRT",
                "TRAIN_TICKET", "WINDOW", "FAN"],
}


# --------------------------------------------------------------------------
# Runtime
# --------------------------------------------------------------------------
VERBOSE: bool = _env("VERBOSE", False)
NUM_THREADS: int = _env("NUM_THREADS", 2)

for _d in (
    DATASETS_DIR, RAW_DIR, PROCESSED_DIR, METADATA_DIR, MODELS_DIR, RESULTS_DIR,
    LOGS_DIR, FEEDBACK_DIR, DOCS_DIR, EVALUATION_DIR,
):
    _d.mkdir(parents=True, exist_ok=True)


def as_dict() -> Dict[str, Any]:
    """All module-level constants (used by ``/model-info`` and the audit)."""
    out: Dict[str, Any] = {}
    for key, value in sorted(globals().items()):
        if key.isupper() and not key.startswith("_") and not key.startswith("ISL_"):
            if isinstance(value, Path):
                out[key] = str(value)
            elif isinstance(value, (str, int, float, bool, list, tuple, dict)) or value is None:
                out[key] = value
    return out


def describe() -> str:  # pragma: no cover - cosmetic
    return (
        f"{PROJECT_NAME}\n"
        f"  dataset        : {DATASET_NAME}\n"
        f"  landmarks      : {NUM_LANDMARKS} ({FEATURE_DIM} features/frame, mask={MASK_ENABLED})\n"
        f"  sequence length: {SEQUENCE_LENGTH}\n"
        f"  seed           : {RANDOM_SEED}\n"
        f"  paths          : {ROOT_DIR}"
    )


if __name__ == "__main__":  # pragma: no cover
    print(describe())
