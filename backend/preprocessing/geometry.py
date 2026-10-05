"""
Geometry calibration for landmark data.

Why this module exists
----------------------
The official AI4Bharat keypoint release (Zenodo 6674324) was produced with a
frame resize whose width/height arguments were swapped (1920x1080 -> 1080x1920),
so the archived **x axis is compressed relative to y** by a constant factor.
Measured on the dataset itself:

    shoulder-width / shoulder-to-hip ratio in the release : 0.2018  (median, n=197, IQR 0.197-0.205)
    same ratio measured by MediaPipe on real full-body photos : 0.6566
    => implied x-axis scale 3.26
    nose->eye-line / inter-ocular in the release : 1.5694
    same ratio measured by MediaPipe                     : 0.5238
    => implied x-axis scale 3.00
    theoretical value from the swap, (W/H)^2 = (1920/1080)^2 : 3.1605

Training on the uncorrected data would teach the model a squashed human body and
would **not** transfer to live MediaPipe frames, which are isotropic.  Therefore
the correction is applied at load time (config.APPLY_KEYPOINT_AXIS_CORRECTION)
and the numbers above are re-measured by ``training/calibrate_geometry.py`` and
reported in ``docs/dataset_audit.md``.

Nothing here is a fudge factor for accuracy: it is a units fix, it is derived
from the data, and it is disabled for the live MediaPipe path (which is already
isotropic).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

import config

logger = logging.getLogger(__name__)

CALIBRATION_FILE = Path(config.METADATA_DIR) / "geometry_calibration.json"

# BlazePose indices used by the reference measurements
NOSE, LEFT_EYE, RIGHT_EYE = 0, 2, 5
L_SHOULDER, R_SHOULDER, L_HIP, R_HIP = 11, 12, 23, 24


# --------------------------------------------------------------------------
# The units fix
# --------------------------------------------------------------------------
def apply_x_axis_scale(coords: np.ndarray, scale: float) -> np.ndarray:
    """Multiply the x channel by ``scale`` (in place on a copy).

    ``coords`` is (..., 3) with (x, y, z) on the last axis.  z is left untouched:
    MediaPipe's z is already reported on the same scale as x for the *original*
    isotropic frame, and the released archive carries it un-resized.
    """
    if scale is None or float(scale) == 1.0:
        return coords
    out = np.asarray(coords, dtype=np.float32).copy()
    out[..., 0] *= float(scale)
    return out


def axis_scale_for_video_shape(vid_shape: Sequence[int]) -> float:
    """Theoretical correction from the archived video shape.

    The release stores ``vid_shape`` as (height, width).  The swap-resize
    hypothesis predicts ``scale = (width / height) ** 2``.
    """
    h, w = int(vid_shape[0]), int(vid_shape[1])
    if h <= 0 or w <= 0:
        return 1.0
    return float((w / h) ** 2)


def correct_landmarks(coords: np.ndarray, vid_shape: Optional[Sequence[int]] = None,
                      scale: Optional[float] = None,
                      enabled: Optional[bool] = None) -> np.ndarray:
    """Apply the source-specific units fix to landmark coordinates.

    Priority: explicit ``scale`` > per-video theoretical value from ``vid_shape``
    (``(w/h)**2``) > ``config.X_AXIS_SCALE``.
    """
    enabled = config.APPLY_KEYPOINT_AXIS_CORRECTION if enabled is None else enabled
    if not enabled:
        return coords
    if scale is not None:
        s = float(scale)
    elif vid_shape is not None:
        s = axis_scale_for_video_shape(vid_shape)
    else:
        s = float(config.X_AXIS_SCALE)
    return apply_x_axis_scale(coords, s)


# --------------------------------------------------------------------------
# Measurement (calibration)
# --------------------------------------------------------------------------
@dataclass
class GeometryMeasurement:
    shoulder_torso_ratio: float = float("nan")
    face_vertical_over_interocular: float = float("nan")
    videos_used: int = 0
    frames_used: int = 0
    x_axis_scale_from_body: float = float("nan")
    x_axis_scale_from_face: float = float("nan")
    x_axis_scale_theoretical: float = float("nan")
    adopted_x_axis_scale: float = float("nan")
    reference: Dict[str, float] = None  # type: ignore[assignment]

    def as_dict(self) -> Dict:
        d = asdict(self)
        d["reference"] = self.reference or dict(config.MEDIAPIPE_REFERENCE)
        return d


def measure_ratios_batch(
    sequences: Iterable[Tuple[np.ndarray, np.ndarray]],
    corrected: bool = False,
) -> GeometryMeasurement:
    """Measure the two anatomical ratios over an iterable of (coords, mask) clips.

    Parameters
    ----------
    sequences : iterable of (T, 75, 3) coordinates and (T, 75) boolean/float masks
    corrected : apply the x-axis fix before measuring (should be False for
        calibration, True only for reporting the post-fix value)
    """
    body, face = [], []
    vids = 0
    frames = 0
    for coords, mask in sequences:
        coords = np.asarray(coords, dtype=np.float32)
        mask = np.asarray(mask) > 0
        if coords.ndim != 3 or coords.shape[0] == 0:
            continue
        vids += 1
        b, f = [], []
        for t in range(coords.shape[0]):
            m = mask[t]
            frames += 1
            if m[L_SHOULDER] and m[R_SHOULDER] and m[L_HIP] and m[R_HIP]:
                sh = abs(float(coords[t, L_SHOULDER, 0] - coords[t, R_SHOULDER, 0]))
                tor = abs(float(coords[t, L_SHOULDER, 1] - coords[t, L_HIP, 1]))
                if tor > 1e-6:
                    b.append(sh / tor)
            if m[NOSE] and m[LEFT_EYE] and m[RIGHT_EYE]:
                e = abs(float(coords[t, LEFT_EYE, 0] - coords[t, RIGHT_EYE, 0]))
                v = abs(float(coords[t, LEFT_EYE, 1] + coords[t, RIGHT_EYE, 1]) / 2.0
                        - float(coords[t, NOSE, 1]))
                if e > 1e-6:
                    f.append(v / e)
        if len(b) >= 5:
            body.append(float(np.median(b)))
        if len(f) >= 5:
            face.append(float(np.median(f)))

    meas = GeometryMeasurement(videos_used=vids, frames_used=frames)
    ref = dict(config.MEDIAPIPE_REFERENCE)
    if body:
        meas.shoulder_torso_ratio = float(np.median(body))
        meas.x_axis_scale_from_body = float(ref["shoulder_torso_ratio"] / meas.shoulder_torso_ratio)
    if face:
        meas.face_vertical_over_interocular = float(np.median(face))
        # NOTE the direction: this ratio is *vertical / horizontal*, so anisotropic
        # x-compression makes it LARGER, and the implied scale is measured/reference
        # (the opposite of the body ratio, which is horizontal / vertical).
        meas.x_axis_scale_from_face = float(meas.face_vertical_over_interocular
                                            / ref["face_vertical_over_interocular"])
    meas.x_axis_scale_theoretical = float(config.X_AXIS_SCALE)
    meas.adopted_x_axis_scale = float(config.X_AXIS_SCALE)
    meas.reference = ref
    return meas


def measure_ratios_from_dataset(limit: Optional[int] = None, seed: int = config.RANDOM_SEED,
                                corrected: bool = False) -> GeometryMeasurement:
    """Calibrate on a random sample of official keypoint files."""
    import random

    from .include50 import load_split_metadata, read_keypoint_archive_file, list_keypoint_files

    files = list_keypoint_files()
    if not files:
        raise FileNotFoundError(
            "No keypoint files found. Set ISL_KEYPOINTS_PATH or run "
            "`python training/fetch_dataset.py --extract`."
        )
    rng = random.Random(seed)
    sample = files if not limit else rng.sample(files, min(limit, len(files)))

    def gen():
        for f in sample:
            try:
                coords, mask, _ = read_keypoint_archive_file(f, apply_axis_fix=corrected)
            except Exception:
                continue
            if coords.shape[0]:
                yield coords, mask

    return measure_ratios_batch(gen(), corrected=corrected)


def write_calibration(meas: GeometryMeasurement, path: Path = CALIBRATION_FILE) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meas.as_dict(), indent=2))
    logger.info("geometry calibration written to %s", path)
    return path


def load_calibration(path: Path = CALIBRATION_FILE) -> Dict:
    if Path(path).exists():
        return json.loads(Path(path).read_text())
    return {}


def describe() -> str:  # pragma: no cover
    return (
        f"x-axis scale: {config.X_AXIS_SCALE:.4f} "
        f"(correction {'ON' if config.APPLY_KEYPOINT_AXIS_CORRECTION else 'OFF'})"
    )
