"""
MediaPipe landmark extraction (pose + left hand + right hand).

Two backends are supported and selected automatically:

1. **Tasks API** (``mediapipe >= 0.10.18`` and ``mediapipe 1.x``) using the
   ``holistic_landmarker.task`` bundle.  This is the only backend available on
   MediaPipe 1.0.1 / Python 3.13, which is what this project runs on
   (``mediapipe.solutions`` was removed in 1.x).
2. **Legacy Solutions API** (``mediapipe.solutions.holistic``) when present.

Both produce exactly the same :class:`~backend.preprocessing.landmarks.FrameLandmarks`
object, so the rest of the system is backend-agnostic.

Usage
-----
    from backend.preprocessing.mediapipe_extractor import MediaPipeExtractor

    with MediaPipeExtractor() as ex:
        frame = ex.extract_frame(rgb_uint8)          # FrameLandmarks
        seq, coords, stats = ex.extract_video("clip.mp4")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

import config
from .landmarks import FrameLandmarks, pack_frame

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ASSET = config.MODELS_DIR / "mediapipe_assets" / "holistic_landmarker.task"

#: MediaPipe Hands landmark count / BlazePose landmark count
N_HAND = 21
N_POSE = 33
N_FACE = 468


def _to_xyz(landmarks: Iterable) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a MediaPipe landmark list into (N,3) coords and (N,) visibility."""
    pts = list(landmarks)
    coords = np.array([[float(p.x), float(p.y), float(p.z)] for p in pts], dtype=np.float32)
    vis = np.array([float(getattr(p, "visibility", 1.0) or 0.0) for p in pts], dtype=np.float32)
    return coords, vis


@dataclass
class ExtractionStats:
    """Counters describing how well MediaPipe did on a video."""

    video: str = ""
    frames_read: int = 0
    frames_decode_error: int = 0
    frames_with_pose: int = 0
    frames_with_left_hand: int = 0
    frames_with_right_hand: int = 0
    frames_with_any_hand: int = 0
    frames_with_nothing: int = 0
    source_fps: float = 0.0
    width: int = 0
    height: int = 0
    duration_s: float = 0.0
    error: str = ""

    @property
    def extraction_success(self) -> float:
        """Fraction of decoded frames with at least one usable landmark group."""
        usable = self.frames_read - self.frames_with_nothing
        return float(usable) / float(self.frames_read) if self.frames_read else 0.0

    @property
    def both_hands_rate(self) -> float:
        both = 0 if not self.frames_read else min(self.frames_with_left_hand, self.frames_with_right_hand)
        return float(both) / float(self.frames_read) if self.frames_read else 0.0

    def as_dict(self) -> Dict[str, float]:
        d = self.__dict__.copy()
        d["extraction_success"] = self.extraction_success
        d["both_hands_rate"] = self.both_hands_rate
        return d


class MediaPipeExtractor:
    """Thin, defensive wrapper around MediaPipe holistic landmark extraction."""

    def __init__(
        self,
        model_asset_path: Optional[Path | str] = None,
        backend: str = "auto",
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        min_hand_confidence: float = 0.5,
        delegate: str = "cpu",
    ) -> None:
        self.model_asset_path = Path(model_asset_path or DEFAULT_MODEL_ASSET)
        self.backend = backend
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.min_hand_confidence = min_hand_confidence
        self.delegate = delegate
        self._impl = None
        self._mode = None
        self._timestamp_ms = 0
        self._frame_shape: Optional[Tuple[int, int]] = None
        self._reinit_count = 0
        self._init_backend()

    # ------------------------------------------------------------------
    # backend selection
    # ------------------------------------------------------------------
    def _init_backend(self) -> None:
        errors: List[str] = []
        if self.backend in ("auto", "tasks"):
            try:
                self._init_tasks()
                self._mode = "tasks"
                logger.info("MediaPipe backend: Tasks API (holistic_landmarker.task)")
                return
            except Exception as exc:  # pragma: no cover - depends on install
                errors.append(f"tasks: {exc}")
        if self.backend in ("auto", "legacy"):
            try:
                self._init_legacy()
                self._mode = "legacy"
                logger.info("MediaPipe backend: legacy solutions.holistic")
                return
            except Exception as exc:  # pragma: no cover
                errors.append(f"legacy: {exc}")
        raise RuntimeError(
            "Could not initialise a MediaPipe backend.\n  "
            + "\n  ".join(errors)
            + f"\n  Tasks API needs the model bundle: {self.model_asset_path}\n"
            "  Run:  python scripts/download_models.py"
        )

    def _init_tasks(self) -> None:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        if not self.model_asset_path.exists():
            raise FileNotFoundError(f"model bundle not found: {self.model_asset_path}")
        base_options = mp_python.BaseOptions(
            model_asset_path=str(self.model_asset_path),
            delegate=mp_python.BaseOptions.Delegate.GPU
            if self.delegate == "gpu"
            else mp_python.BaseOptions.Delegate.CPU,
        )
        options = vision.HolisticLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            min_face_detection_confidence=self.min_detection_confidence,
            min_face_landmarks_confidence=self.min_detection_confidence,
            min_pose_detection_confidence=self.min_detection_confidence,
            min_pose_landmarks_confidence=self.min_detection_confidence,
            min_hand_landmarks_confidence=self.min_hand_confidence,
            output_face_blendshapes=False,
            output_segmentation_mask=False,
        )
        self._vision = vision
        self._mp = mp
        self._impl = vision.HolisticLandmarker.create_from_options(options)

    def _init_legacy(self) -> None:  # pragma: no cover - only on mediapipe < 1.0
        import mediapipe as mp

        if not hasattr(mp, "solutions"):
            raise AttributeError("mediapipe.solutions not available in this build")
        self._mp = mp
        self._impl = mp.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )

    def _ensure_frame_shape(self, shape: Tuple[int, int]) -> None:
        """The Tasks API video graph requires a constant frame size.

        Switching camera resolution, or feeding a new video with different
        dimensions, otherwise kills the graph with an INTERNAL error
        (``SegmentationSmoothingCalculator ... current_mat->cols == previous_mat->cols``).
        We transparently rebuild the landmarker when the size changes.
        """
        if self._mode != "tasks" or self._frame_shape is None:
            self._frame_shape = shape
            return
        if shape != self._frame_shape:
            logger.debug("frame size changed %s -> %s; rebuilding landmarker",
                         self._frame_shape, shape)
            self.close()
            self._init_tasks()
            self._frame_shape = shape
            self._reinit_count += 1

    # ------------------------------------------------------------------
    # per-frame extraction
    # ------------------------------------------------------------------
    def extract_frame(self, rgb: np.ndarray, timestamp_ms: Optional[int] = None) -> FrameLandmarks:
        """Extract pose + both hands from one RGB uint8 frame."""
        if rgb is None or rgb.size == 0:
            return FrameLandmarks()
        rgb = np.ascontiguousarray(rgb[:, :, ::-1] if rgb.shape[-1] == 3 and rgb.dtype == np.float32 else rgb)
        self._ensure_frame_shape((int(rgb.shape[0]), int(rgb.shape[1])))

        if self._mode == "tasks":
            image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB,
                                   data=np.ascontiguousarray(rgb.astype(np.uint8)))
            # The Tasks API requires strictly increasing timestamps per graph.
            ts = int(timestamp_ms) if timestamp_ms is not None else self._timestamp_ms
            ts = max(ts, self._timestamp_ms)
            self._timestamp_ms = ts + 1
            res = self._impl.detect_for_video(image, ts)
            pose = getattr(res, "pose_landmarks", None)
            lh = getattr(res, "left_hand_landmarks", None)
            rh = getattr(res, "right_hand_landmarks", None)
            pose_c, pose_v = _to_xyz(pose) if pose else (None, None)
            lh_c = _to_xyz(lh)[0] if lh else None
            rh_c = _to_xyz(rh)[0] if rh else None
        else:  # pragma: no cover - legacy path
            res = self._impl.process(np.ascontiguousarray(rgb))
            pose_c, pose_v = _to_xyz(res.pose_landmarks.landmark) if res.pose_landmarks else (None, None)
            lh_c = _to_xyz(res.left_hand_landmarks.landmark)[0] if res.left_hand_landmarks else None
            rh_c = _to_xyz(res.right_hand_landmarks.landmark)[0] if res.right_hand_landmarks else None

        return FrameLandmarks(pose=pose_c, left_hand=lh_c, right_hand=rh_c, pose_visibility=pose_v)

    # ------------------------------------------------------------------
    # video level
    # ------------------------------------------------------------------
    def extract_video(
        self,
        video_path: Path | str,
        max_frames: Optional[int] = None,
        frame_stride: int = 1,
        sample_every: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, ExtractionStats]:
        """Extract landmarks from a video file.

        Parameters
        ----------
        video_path : path to a video readable by OpenCV
        max_frames : stop after this many *decoded* frames (privacy/speed guard)
        frame_stride : decode every n-th frame (``sample_every`` is an alias)

        Returns
        -------
        coords : (T, 75, 3) float32 in normalised image coordinates
        mask   : (T, 75)   float32 presence mask
        stats  : :class:`ExtractionStats`
        """
        import cv2

        video_path = Path(video_path)
        stride = int(sample_every or frame_stride)
        stats = ExtractionStats(video=str(video_path))
        coords_out: List[np.ndarray] = []
        masks_out: List[np.ndarray] = []

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            stats.error = "video_open_failed"
            return (np.zeros((0, config.NUM_LANDMARKS, config.COORDS_PER_LANDMARK), dtype=np.float32),
                    np.zeros((0, config.NUM_LANDMARKS), dtype=np.float32), stats)

        stats.source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        stats.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        stats.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        total = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        if stats.source_fps > 0:
            stats.duration_s = total / stats.source_fps

        limit = int(max_frames or config.MAX_FRAMES_READ)
        idx = 0
        try:
            while idx < limit:
                ok, frame_bgr = cap.read()
                if not ok:
                    break
                idx += 1
                if stride > 1 and (idx - 1) % stride != 0:
                    continue
                stats.frames_read += 1
                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                ts = int(1000.0 * (idx - 1) / stats.source_fps) if stats.source_fps > 0 else idx * 33
                lm = self.extract_frame(rgb, timestamp_ms=ts)
                c, m = pack_frame(lm)
                coords_out.append(c)
                masks_out.append(m)

                stats.frames_with_pose += int(lm.pose is not None)
                stats.frames_with_left_hand += int(lm.left_hand is not None)
                stats.frames_with_right_hand += int(lm.right_hand is not None)
                stats.frames_with_any_hand += int(lm.left_hand is not None or lm.right_hand is not None)
                if lm.is_empty():
                    stats.frames_with_nothing += 1
        except Exception as exc:  # corrupt file mid-stream
            stats.error = f"{type(exc).__name__}: {exc}"
        finally:
            cap.release()

        if not coords_out:
            return (np.zeros((0, config.NUM_LANDMARKS, config.COORDS_PER_LANDMARK), dtype=np.float32),
                    np.zeros((0, config.NUM_LANDMARKS), dtype=np.float32), stats)
        return (np.stack(coords_out).astype(np.float32), np.stack(masks_out).astype(np.float32), stats)

    def reset(self) -> None:
        """Reset tracking state between videos (keeps the graph alive)."""
        self._timestamp_ms = 0

    def close(self) -> None:
        try:
            if self._impl is not None and hasattr(self._impl, "close"):
                self._impl.close()
        except Exception:  # pragma: no cover
            pass
        self._impl = None

    def __enter__(self) -> "MediaPipeExtractor":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self):  # pragma: no cover
        self.close()


def extract_landmarks_for_frames(
    frames: Iterable[np.ndarray], extractor: Optional[MediaPipeExtractor] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Convenience helper: run extraction over an iterable of RGB frames."""
    own = extractor is None
    ex = extractor or MediaPipeExtractor()
    coords, masks = [], []
    for i, f in enumerate(frames):
        c, m = pack_frame(ex.extract_frame(f, timestamp_ms=i * 33))
        coords.append(c)
        masks.append(m)
    if own:
        ex.close()
    if not coords:
        return (np.zeros((0, config.NUM_LANDMARKS, config.COORDS_PER_LANDMARK), dtype=np.float32),
                np.zeros((0, config.NUM_LANDMARKS), dtype=np.float32))
    return np.stack(coords), np.stack(masks)
