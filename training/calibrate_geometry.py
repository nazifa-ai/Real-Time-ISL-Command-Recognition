#!/usr/bin/env python3
"""
Measure (and optionally re-derive) the axis-units correction for the official
AI4Bharat keypoint release.

Two independent anatomical priors are measured on the dataset and compared with
the same ratios measured by MediaPipe on real photographs (isotropic, in true
pixels):

  body : |left_shoulder - right_shoulder| / |mid_shoulder - mid_hip|
  face : |nose -> mid(eye_line)| / |left_eye - right_eye|

If the archived data were isotropic, both ratios would match MediaPipe directly.
The ratio of the two gives the implied x-axis scale; the theoretical value for a
1920x1080 -> 1080x1920 resize is (W/H)^2 = 3.1605.

Usage
-----
    python training/calibrate_geometry.py                       # dataset only
    python training/calibrate_geometry.py --photos img1.jpg ...  # also re-measure the reference
    python training/calibrate_geometry.py --sample 400
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from backend.preprocessing import geometry as geom  # noqa: E402


def measure_reference(photos, verbose: bool = True) -> dict:
    """Measure the two anatomical ratios with MediaPipe on real photos."""
    import cv2

    from backend.preprocessing.mediapipe_extractor import MediaPipeExtractor

    ex = MediaPipeExtractor()
    body, face, used = [], [], []
    try:
        for i, p in enumerate(photos):
            img = cv2.imread(str(p))
            if img is None:
                print(f"  ! cannot read {p}")
                continue
            h, w = img.shape[:2]
            lm = ex.extract_frame(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 1000 + i * 40)
            if lm.pose is None:
                print(f"  ! no pose detected in {p}")
                continue
            pose = np.asarray(lm.pose, dtype=np.float32)[:33]
            px = np.stack([pose[:, 0] * w, pose[:, 1] * h], axis=-1)  # true pixels
            sh = float(np.linalg.norm(px[11] - px[12]))
            torso = float(np.linalg.norm((px[11] + px[12]) / 2 - (px[23] + px[24]) / 2))
            if torso > 1e-6:
                body.append(sh / torso)
            e = float(abs(px[2, 0] - px[5, 0]))
            v = float(abs((px[2, 1] + px[5, 1]) / 2 - px[0, 1]))
            if e > 1e-6:
                face.append(v / e)
            used.append(str(p))
    finally:
        ex.close()

    out = {
        "photos_used": used,
        "shoulder_torso_ratio": float(np.median(body)) if body else float("nan"),
        "face_vertical_over_interocular": float(np.median(face)) if face else float("nan"),
    }
    if verbose:
        print(f"  reference photos usable: {len(used)}")
        print(f"  shoulder/torso  = {out['shoulder_torso_ratio']:.4f}")
        print(f"  face V/E        = {out['face_vertical_over_interocular']:.4f}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", type=int, default=200, help="videos to measure (default 200)")
    ap.add_argument("--photos", nargs="*", default=None,
                    help="full-body reference photographs (optional; re-measures the reference)")
    ap.add_argument("--adopt", type=float, default=None,
                    help="adopt an explicit X_AXIS_SCALE instead of the measured mean")
    args = ap.parse_args(argv)

    print("1) Measuring INCLUDE keypoints (uncorrected units) ...")
    raw = geom.measure_ratios_from_dataset(limit=args.sample, corrected=False)
    print(f"   videos={raw.videos_used} frames={raw.frames_used}")
    print(f"   shoulder/torso = {raw.shoulder_torso_ratio:.4f}")
    print(f"   face V/E       = {raw.face_vertical_over_interocular:.4f}")
    print(f"   implied scale: body={raw.x_axis_scale_from_body:.3f}  face={raw.x_axis_scale_from_face:.3f}")

    if args.photos:
        print("2) Re-measuring the MediaPipe reference on supplied photos ...")
        ref = measure_reference(args.photos)
        ref = {k: v for k, v in ref.items()
               if k in ("shoulder_torso_ratio", "face_vertical_over_interocular")}
        raw.reference = {**dict(config.MEDIAPIPE_REFERENCE), **ref}
        if ref.get("shoulder_torso_ratio"):
            raw.x_axis_scale_from_body = ref["shoulder_torso_ratio"] / raw.shoulder_torso_ratio
        if ref.get("face_vertical_over_interocular"):
            raw.x_axis_scale_from_face = ref["face_vertical_over_interocular"] / raw.face_vertical_over_interocular

    measured = [v for v in (raw.x_axis_scale_from_body, raw.x_axis_scale_from_face) if v == v]
    if args.adopt is not None:
        adopted = float(args.adopt)
    elif measured:
        adopted = float(np.mean(measured))
    else:
        adopted = float(config.X_AXIS_SCALE)
    raw.adopted_x_axis_scale = adopted
    raw.x_axis_scale_theoretical = float((1920.0 / 1080.0) ** 2)

    print("3) Cross-check against the theoretical resize value")
    print(f"   theoretical (1920/1080)^2 = {raw.x_axis_scale_theoretical:.4f}")
    print(f"   measured mean             = {adopted:.4f}")
    print(f"   difference                = {100.0*abs(adopted-raw.x_axis_scale_theoretical)/raw.x_axis_scale_theoretical:.1f}%")

    out = geom.write_calibration(raw)
    print(f"\nWritten: {out}")
    print("config.X_AXIS_SCALE stays at the theoretical value unless you set "
          "ISL_X_AXIS_SCALE explicitly; both agree within the spread of the priors.")
    print(json.dumps({k: v for k, v in raw.as_dict().items() if "reference" not in k}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
