"""
From raw RGB frames to model input, using exactly the preprocessing chain that
training used (``backend.preprocessing.landmarks`` + ``backend.preprocessing.sequence``).

Used by the webcam loop, the batch video runner and the API's video endpoint, so
the live path and the training path cannot diverge.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Sequence

import numpy as np

import config
from backend.preprocessing import landmarks as lm
from backend.preprocessing import sequence as seq_ops
from backend.preprocessing.mediapipe_extractor import MediaPipeExtractor


@dataclass
class BufferStats:
    frames_seen: int = 0
    frames_with_pose: int = 0
    frames_with_left: int = 0
    frames_with_right: int = 0
    frames_with_nothing: int = 0
    started_at: float = field(default_factory=time.time)

    @property
    def elapsed(self) -> float:
        return max(1e-6, time.time() - self.started_at)

    @property
    def fps(self) -> float:
        return self.frames_seen / self.elapsed

    def as_dict(self) -> Dict[str, float]:
        return {
            "frames_seen": self.frames_seen,
            "pose_rate": self.frames_with_pose / max(1, self.frames_seen),
            "left_hand_rate": self.frames_with_left / max(1, self.frames_seen),
            "right_hand_rate": self.frames_with_right / max(1, self.frames_seen),
            "empty_rate": self.frames_with_nothing / max(1, self.frames_seen),
            "fps": self.fps,
        }


class FrameBuffer:
    """Rolling window of landmark frames -> fixed-length feature sequence.

    Landmarks are normalised *per frame at insertion time*, so the tensor handed
    to the model is built the same way whether the frames came from a webcam, an
    uploaded video or the dataset.
    """

    def __init__(self, length: Optional[int] = None, extractor: Optional[MediaPipeExtractor] = None,
                 stride: int = 1):
        self.length = int(length or config.SEQUENCE_LENGTH)
        self.stride = max(1, int(stride))
        self.extractor = extractor or MediaPipeExtractor()
        self._frames: Deque[np.ndarray] = deque(maxlen=self.length)
        self.last_landmarks = None
        self.last_stage_timings = {"mediapipe_ms": 0.0, "preprocessing_ms": 0.0, "buffer_ms": 0.0}
        self.stats = BufferStats()
        self._n_since_push = 0

    # ------------------------------------------------------------------
    def push(self, rgb: np.ndarray, timestamp_ms: Optional[int] = None) -> Optional[np.ndarray]:
        """Add one frame; returns ``(length, 225)`` features once the window is full."""
        push_started = time.perf_counter()
        self.stats.frames_seen += 1
        stage_started = time.perf_counter()
        landmarks = self.extractor.extract_frame(rgb, timestamp_ms=timestamp_ms)
        mediapipe_ms = (time.perf_counter() - stage_started) * 1000.0
        self.last_landmarks = landmarks
        self.stats.frames_with_pose += int(landmarks.pose is not None)
        self.stats.frames_with_left += int(landmarks.left_hand is not None)
        self.stats.frames_with_right += int(landmarks.right_hand is not None)
        self.stats.frames_with_nothing += int(landmarks.is_empty())

        stage_started = time.perf_counter()
        coords, mask = lm.pack_frame(landmarks)
        feats, _ = lm.normalise_sequence(coords[None, ...], mask[None, ...])
        preprocessing_ms = (time.perf_counter() - stage_started) * 1000.0
        stage_started = time.perf_counter()
        self._frames.append(feats[0])

        self._n_since_push += 1
        buffering_ms = (time.perf_counter() - stage_started) * 1000.0
        self.last_stage_timings = {
            "mediapipe_ms": round(mediapipe_ms, 3),
            "preprocessing_ms": round(preprocessing_ms, 3),
            "buffer_ms": round(buffering_ms, 3),
            "frame_total_ms": round((time.perf_counter() - push_started) * 1000.0, 3),
        }
        if len(self._frames) < self.length or (self._n_since_push % self.stride):
            return None
        return self.features()

    # ------------------------------------------------------------------
    def features(self) -> Optional[np.ndarray]:
        if len(self._frames) < self.length:
            return None
        return np.stack(self._frames).astype(np.float32)

    def ready(self) -> bool:
        return len(self._frames) >= self.length

    def fill_ratio(self) -> float:
        return len(self._frames) / self.length

    def reset(self) -> None:
        self._frames.clear()
        self._n_since_push = 0
        self.last_landmarks = None

    def close(self) -> None:
        self.extractor.close()


# --------------------------------------------------------------------------
# Offline helpers (batch inference without a camera)
# --------------------------------------------------------------------------
def features_from_rgb_frames(frames: Sequence[np.ndarray], extractor: MediaPipeExtractor,
                             length: Optional[int] = None, return_landmarks: bool = False):
    """Extract + normalise + resample a whole clip of RGB frames.

    When ``return_landmarks=True``, also returns the raw (pre-normalisation)
    per-frame landmark coordinates, resampled with the exact same method to
    the same ``length`` timesteps as the features -- i.e. these are literally
    the landmarks behind the sequence the model consumes, not a separate
    estimate, so a caller can honestly visualise "what the model saw".
    """
    length = int(length or config.SEQUENCE_LENGTH)
    coords, masks = [], []
    for i, f in enumerate(frames):
        c, m = lm.pack_frame(extractor.extract_frame(f, timestamp_ms=i * 33))
        coords.append(c)
        masks.append(m)
    if not coords:
        return (None, None) if return_landmarks else None
    coords = np.stack(coords)
    mask = np.stack(masks)
    usable = (mask > 0).any(axis=1)
    if int(usable.sum()) < config.MIN_VALID_FRAMES:
        return (None, None) if return_landmarks else None
    feats, _ = lm.normalise_sequence(coords, mask)
    feats = seq_ops.resample_sequence(feats, length, method=config.SEQ_RESAMPLE_METHOD)
    if not return_landmarks:
        return feats
    t, n, c = coords.shape
    coords_rs = seq_ops.resample_sequence(coords.reshape(t, n * c), length,
                                          method=config.SEQ_RESAMPLE_METHOD).reshape(length, n, c)
    return feats, coords_rs


def features_from_video(path, extractor: Optional[MediaPipeExtractor] = None,
                        max_frames: Optional[int] = None,
                        length: Optional[int] = None, return_landmarks: bool = False):
    """Decode a video file and return (features, stats) for one clip.

    When ``return_landmarks=True``, ``stats["landmarks"]`` carries the raw
    per-frame coordinates behind the returned feature sequence (see
    ``features_from_rgb_frames``), as a plain nested list ready for JSON.
    """
    import cv2

    own = extractor is None
    ex = extractor or MediaPipeExtractor()
    stats = None
    frames: List[np.ndarray] = []
    try:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return None, {"error": "video_open_failed", "path": str(path)}
        n = 0
        limit = int(max_frames or config.MAX_FRAMES_READ)
        while n < limit:
            ok, frame = cap.read()
            if not ok:
                break
            n += 1
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        if not frames:
            return None, {"error": "no_frames_decoded", "path": str(path)}
        stats = {"path": str(path), "frames_decoded": len(frames)}
        if return_landmarks:
            feats, coords_rs = features_from_rgb_frames(frames, ex, length=length, return_landmarks=True)
            if coords_rs is not None:
                stats["landmarks"] = coords_rs.tolist()
        else:
            feats = features_from_rgb_frames(frames, ex, length=length)
        return feats, stats
    finally:
        if own:
            ex.close()


def dataset_features(filepath: str, length: Optional[int] = None) -> Optional[np.ndarray]:
    """Features for one dataset clip (used for demo/verification without a camera)."""
    from backend.preprocessing import include50

    coords, mask, _ = include50.read_keypoints_for(filepath)
    if coords.shape[0] == 0:
        return None
    usable = (mask > 0).any(axis=1)
    if int(usable.sum()) < config.MIN_VALID_FRAMES:
        return None
    feats, _ = lm.normalise_sequence(coords, mask)
    return seq_ops.resample_sequence(feats, int(length or config.SEQUENCE_LENGTH))
