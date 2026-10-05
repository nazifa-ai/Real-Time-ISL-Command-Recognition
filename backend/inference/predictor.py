"""
Prediction engine shared by real-time inference, batch inference and the API.

Pipeline implemented here (identical in every entry point, so behaviour cannot
drift between the demo and the service):

    features (T, 225)
        -> trained model -> softmax probabilities
        -> confidence gate          (>= CONFIDENCE_THRESHOLD -> CONFIDENT, else UNCERTAIN)
        -> temporal smoothing       (sliding-window majority vote)
        -> stable-prediction check  (STABLE_FRAMES_REQUIRED consecutive agreements)
        -> duplicate suppression    (a held sign is not re-emitted)
        -> caption update + optional speech decision
"""

from __future__ import annotations

import time
from collections import Counter, deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

import config


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------
@dataclass
class PredictionResult:
    sign: str                       # smoothed/gated label: class name, UNCERTAIN or IDLE
    confidence: float               # confidence of the smoothed decision
    status: str                     # CONFIDENT | UNCERTAIN | PENDING | IDLE
    latency_ms: float
    model: str
    model_latency_ms: float = 0.0
    smoothing_latency_ms: float = 0.0
    hand_gate_latency_ms: float = 0.0
    raw_sign: str = ""              # argmax of this frame, before gating/smoothing
    raw_confidence: float = 0.0
    stable: bool = False
    streak: int = 0
    should_speak: bool = False
    caption: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    top_k: List[Dict[str, object]] = field(default_factory=list)

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


# --------------------------------------------------------------------------
# Temporal smoothing
# --------------------------------------------------------------------------
class TemporalSmoother:
    """Sliding-window majority vote + stable-prediction detection.

    A held sign must not be shouted repeatedly ("HELLO HELLO HELLO"), so an
    emission is produced only when the smoothed label has been stable for
    ``stable_frames`` updates *and* differs from the last emitted sign (or the
    speech cooldown has expired).
    """

    def __init__(self, window: int = config.SMOOTHING_WINDOW,
                 min_votes: int = config.SMOOTHING_MIN_VOTES,
                 stable_frames: int = config.STABLE_FRAMES_REQUIRED):
        self.window: Deque[Tuple[str, float]] = deque(maxlen=max(1, window))
        self.min_votes = max(1, min(min_votes, max(1, window)))
        self.stable_frames = max(1, stable_frames)
        self._streak = 0
        self._streak_label = ""
        self._last_emitted = ""
        self._last_emitted_at = 0.0

    def reset(self) -> None:
        self.window.clear()
        self._streak = 0
        self._streak_label = ""
        self._last_emitted = ""
        self._last_emitted_at = 0.0

    def update(self, label: str, confidence: float, now: Optional[float] = None) -> Dict[str, object]:
        self.window.append((label, float(confidence)))
        votes = Counter(l for l, _ in self.window)
        top_label, top_votes = votes.most_common(1)[0]
        smoothed_conf = float(np.mean([c for l, c in self.window if l == top_label]))
        majority = top_votes >= self.min_votes

        candidate = top_label if majority else config.UNCERTAIN_LABEL

        if candidate == self._streak_label:
            self._streak += 1
        else:
            self._streak_label = candidate
            self._streak = 1

        is_stable = (candidate not in (config.UNCERTAIN_LABEL, config.IDLE_LABEL)
                     and self._streak >= self.stable_frames)

        now = time.time() if now is None else now
        is_new = candidate != self._last_emitted
        cooldown_expired = (now - self._last_emitted_at) >= config.TTS_COOLDOWN
        should_emit = is_stable and (is_new or (not config.DUPLICATE_SUPPRESSION) or cooldown_expired)
        if should_emit:
            self._last_emitted = candidate
            self._last_emitted_at = now

        return {
            "label": candidate,
            "confidence": smoothed_conf,
            "votes": top_votes,
            "window": len(self.window),
            "streak": self._streak,
            "stable": is_stable,
            "should_emit": should_emit,
            "is_new": is_new,
        }


# --------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------
class PredictionEngine:
    """Wraps a trained Keras model with gating, smoothing, captioning and speech."""

    def __init__(self, model_name: str = config.DEFAULT_MODEL, model=None,
                 classes: Optional[Sequence[str]] = None, threshold: Optional[float] = None,
                 smoother: Optional[TemporalSmoother] = None, load: bool = True,
                 vocab_filter: Optional[Sequence[str]] = None,
                 weights_path: Optional[Path] = None):
        self.model_name = model_name
        self.threshold = float(config.CONFIDENCE_THRESHOLD if threshold is None else threshold)
        self.smoother = smoother or TemporalSmoother()
        self._speaker = None
        self.speech_enabled = config.TTS_ENABLED_DEFAULT
        self.caption: List[str] = []
        self._last_speak_at = 0.0
        self._last_spoken = ""
        self.latencies: Deque[float] = deque(maxlen=200)
        self.weights_path = Path(weights_path) if weights_path else None
        self.vocab_filter: Optional[List[str]] = None

        if classes is None:
            from backend.preprocessing.include50 import load_vocabulary

            classes = load_vocabulary()
            if not classes:
                raise RuntimeError("No vocabulary found: run training/fetch_metadata.py")
        self.classes = list(classes)

        if model is None and load:
            self.model = self._load_model(model_name)
        else:
            self.model = model

        if vocab_filter:
            self.set_vocab_filter(vocab_filter)

    # -- loading -------------------------------------------------------
    @property
    def model_path(self) -> Path:
        if self.weights_path is not None:
            return self.weights_path
        return Path(config.MODEL_PATHS[self.model_name]) / "model.keras"

    def set_vocab_filter(self, allowed: Optional[Sequence[str]]) -> List[str]:
        """Restrict decoding to a vocabulary subset (personalisation profiles).

        Classes outside the subset get probability 0 and the remainder is
        renormalised; the model itself is untouched.  Passing ``None`` clears it.
        """
        if not allowed:
            self.vocab_filter = None
            return []
        upper = {str(c).strip().upper() for c in allowed if str(c).strip()}
        kept = [c for c in self.classes if c.upper() in upper]
        self.vocab_filter = kept or None
        return kept

    def _load_model(self, model_name: str):
        import tensorflow as tf
        import json

        path = self.weights_path or (Path(config.MODEL_PATHS[model_name]) / "model.keras")
        if not path.exists():
            raise FileNotFoundError(
                f"trained model not found: {path}\n"
                f"Run: python training/train_{model_name}.py"
            )
        model = tf.keras.models.load_model(str(path))
        input_shape = tuple(model.input_shape)
        # The trained Keras builders intentionally declare a dynamic time axis
        # (None, None, D); live inference still supplies the fixed 30-frame
        # contract. Accept that declaration while validating both the feature
        # width and any explicitly fixed time dimension.
        time_dim = input_shape[1] if len(input_shape) == 3 else None
        feature_dim = input_shape[2] if len(input_shape) == 3 else None
        if (len(input_shape) != 3
                or time_dim not in (None, config.SEQUENCE_LENGTH)
                or feature_dim != config.MODEL_INPUT_DIM):
            raise ValueError(
                f"{model_name} input shape {input_shape} is incompatible with the trained feature contract "
                f"(batch, {config.SEQUENCE_LENGTH}, {config.MODEL_INPUT_DIM}); "
                "a dynamic time axis is allowed"
            )
        output_shape = tuple(model.output_shape)
        if len(output_shape) != 2 or output_shape[-1] != len(self.classes):
            raise ValueError(
                f"{model_name} output shape {output_shape} does not match {len(self.classes)} vocabulary labels"
            )
        mapping_path = Path(config.MODEL_PATHS[model_name]) / "class_mapping.json"
        if mapping_path.exists():
            mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
            if mapping.get("classes") != self.classes:
                raise ValueError(f"{model_name} class_mapping.json does not match the active vocabulary order")
        return model

    def switch_model(self, model_name: str) -> None:
        if model_name == self.model_name:
            return
        self.model_name = model_name
        self.model = self._load_model(model_name)
        self.smoother.reset()

    # -- speech --------------------------------------------------------
    @property
    def speaker(self):
        if self._speaker is None:
            from backend.inference.audio import Speaker

            self._speaker = Speaker()
        return self._speaker

    def set_speech(self, enabled: bool) -> None:
        self.speech_enabled = bool(enabled)

    def speak_label(self, label: str) -> bool:
        """Speak one already-gated clip decision, bypassing frame-stream smoothing.

        Uploaded clips are classified once as complete temporal sequences. They
        must not be passed through the live-frame stability gate a second time.
        """
        if not str(label).strip():
            return False
        self._speak(str(label))
        return True

    # -- inference -----------------------------------------------------
    def probabilities(self, features: np.ndarray) -> np.ndarray:
        """(T, D) or (1, T, D) -> (num_classes,) softmax probabilities."""
        x = np.asarray(features, dtype=np.float32)
        if x.ndim == 2:
            x = x[None, ...]
        # Keras `predict()` spins up a data adapter for each single webcam
        # sequence. `predict_on_batch()` uses the same model path without that
        # per-frame adapter overhead (benchmarked on the shipped CPU models).
        probs = self.model.predict_on_batch(x)[0]
        probs = np.asarray(probs, dtype=np.float64)
        if self.vocab_filter:
            allowed = {i for i, c in enumerate(self.classes) if c in self.vocab_filter}
            mask = np.zeros_like(probs)
            for i in allowed:
                mask[i] = probs[i]
            total = float(mask.sum())
            if total > 0:
                probs = mask / total
        return probs

    @staticmethod
    def _hand_activity(features: np.ndarray) -> Tuple[float, float]:
        """(presence_ratio, motion) over the hand-landmark columns of a (T, D) window.

        presence_ratio: fraction of frames with at least one non-zero hand landmark.
        motion: mean absolute frame-to-frame change within the hand columns -- near
        zero means the hands are present but held still (resting), not signing.
        """
        x = np.asarray(features, dtype=np.float32)
        if x.ndim != 2 or x.shape[0] == 0:
            return 0.0, 0.0
        hand_cols = slice(config.LAYOUT.slice("left_hand").start * config.LAYOUT.coords,
                           config.LAYOUT.slice("right_hand").stop * config.LAYOUT.coords)
        hands = x[:, hand_cols]
        per_frame_present = np.abs(hands).sum(axis=1) > 1e-4
        presence_ratio = float(per_frame_present.mean())
        motion = float(np.abs(np.diff(hands, axis=0)).mean()) if hands.shape[0] > 1 else 0.0
        return presence_ratio, motion

    def predict(self, features: np.ndarray, speak: Optional[bool] = None,
                fps: Optional[float] = None) -> PredictionResult:
        t0 = time.perf_counter()

        gate_started = time.perf_counter()
        presence_ratio, motion = self._hand_activity(features)
        idle = (presence_ratio < config.IDLE_MIN_HAND_PRESENCE_RATIO
                or motion < config.IDLE_MIN_HAND_MOTION)

        if idle:
            probs = np.zeros(len(self.classes), dtype=np.float64)
            idx, raw_conf, raw_label = -1, 0.0, config.IDLE_LABEL
            gated = config.IDLE_LABEL
        else:
            model_started = time.perf_counter()
            probs = self.probabilities(features)
            model_latency_ms = (time.perf_counter() - model_started) * 1000.0
            idx = int(np.argmax(probs))
            raw_conf = float(probs[idx])
            raw_label = self.classes[idx] if idx < len(self.classes) else str(idx)
            gated = raw_label if raw_conf >= self.threshold else config.UNCERTAIN_LABEL

        if idle:
            model_latency_ms = 0.0
        hand_gate_latency_ms = (time.perf_counter() - gate_started) * 1000.0

        top_k: List[Dict[str, object]] = []
        if not idle:
            for top_idx in np.argsort(probs)[::-1][:5]:
                top_k.append({"class": self.classes[int(top_idx)],
                              "probability": float(probs[int(top_idx)])})

        smoothing_started = time.perf_counter()
        smooth = self.smoother.update(gated, raw_conf)
        smoothing_latency_ms = (time.perf_counter() - smoothing_started) * 1000.0
        latency_ms = (time.perf_counter() - t0) * 1000.0
        self.latencies.append(latency_ms)

        label = str(smooth["label"])
        confidence = float(smooth["confidence"])
        if label == config.IDLE_LABEL:
            status = config.IDLE_LABEL
        elif label == config.UNCERTAIN_LABEL:
            status = config.UNCERTAIN_LABEL
        elif smooth["stable"]:
            status = "CONFIDENT"
        else:
            status = "PENDING"

        should_speak = bool(smooth["should_emit"] and status == "CONFIDENT")
        if should_speak:
            self.caption.append(label)
        if should_speak and (speak if speak is not None else self.speech_enabled):
            self._speak(label)

        return PredictionResult(
            sign=label, confidence=confidence, status=status, latency_ms=latency_ms,
            model=self.model_name, model_latency_ms=model_latency_ms,
            smoothing_latency_ms=smoothing_latency_ms,
            hand_gate_latency_ms=hand_gate_latency_ms,
            raw_sign=raw_label, raw_confidence=raw_conf,
            stable=bool(smooth["stable"]), streak=int(smooth["streak"]),
            should_speak=should_speak, caption=" ".join(self.caption), top_k=top_k,
        )

    def _speak(self, label: str) -> None:
        now = time.time()
        if label == self._last_spoken and (now - self._last_speak_at) < config.TTS_COOLDOWN:
            return
        try:
            self.speaker.say(label.replace("_", " "))
            self._last_spoken = label
            self._last_speak_at = now
        except Exception:  # TTS must never break recognition
            pass

    # -- caption -------------------------------------------------------
    def reset_caption(self) -> None:
        self.caption.clear()

    def pop_last(self) -> Optional[str]:
        return self.caption.pop() if self.caption else None

    def clear_window(self) -> None:
        self.smoother.reset()

    # -- telemetry -----------------------------------------------------
    def stats(self) -> Dict[str, object]:
        lat = list(self.latencies)
        return {
            "model": self.model_name,
            "threshold": self.threshold,
            "mean_latency_ms": float(np.mean(lat)) if lat else 0.0,
            "p95_latency_ms": float(np.percentile(lat, 95)) if lat else 0.0,
            "throughput_fps": float(1000.0 / np.mean(lat)) if lat else 0.0,
            "caption": " ".join(self.caption),
            "caption_length": len(self.caption),
        }


def load_engine(model_name: Optional[str] = None, threshold: Optional[float] = None) -> PredictionEngine:
    """Convenience factory used by the CLI, the API and the tests."""
    return PredictionEngine(model_name=model_name or config.DEFAULT_MODEL, threshold=threshold)
