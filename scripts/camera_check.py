#!/usr/bin/env python3
"""
Webcam readiness check.

Answers "why is the camera not working?" with measurements instead of guesses:

    python scripts/camera_check.py             # or: python run.py camera
    python scripts/camera_check.py --indices 0 1 2 3 4 --capture md_out.txt

Exit codes: 0 = a camera is usable, 1 = no camera (with the reason printed),
2 = no camera *and* no display (i.e. this is a headless machine).

The check is read-only: it opens each device briefly, reads two frames, releases it,
and (optionally) runs the MediaPipe extractor on one frame to confirm the whole
capture -> landmarks path works here.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from backend.inference import camera as camera_utils  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--indices", nargs="*", type=int, default=None,
                    help=f"indices to test (default 0..{config.CAMERA_PROBE_MAX_INDEX})")
    ap.add_argument("--no-extractor", action="store_true",
                    help="skip the MediaPipe end-to-end check")
    args = ap.parse_args(argv)

    rep = camera_utils.report(indices=args.indices)

    print("Webcam readiness check")
    print("=" * 72)
    print(f"  platform            : {rep.platform}")
    print(f"  capture devices     : {rep.device_nodes or 'none (/dev/video* absent)'}")
    print(f"  DISPLAY             : {rep.display or '<unset>'}")
    print(f"  OpenCV GUI backend  : {rep.opencv_gui_backend}")
    print(f"  preview window      : {rep.gui_available}  ({rep.gui_note})")
    print("  capture probes:")
    for probe in rep.probes:
        mark = "OK   " if probe.readable else "--   "
        print(f"    [{mark}] index {probe.index}: opened={probe.opened} readable={probe.readable} "
              f"shape={probe.shape} fps={probe.fps:.1f} {probe.error}")

    if rep.usable and not args.no_extractor:
        print("  MediaPipe end-to-end check:")
        try:
            import cv2

            from backend.preprocessing.mediapipe_extractor import MediaPipeExtractor

            cap, _, _ = camera_utils.open_camera(preferred_index=rep.usable_index)
            assert cap is not None
            ok, frame = cap.read()
            cap.release()
            if ok:
                extractor = MediaPipeExtractor()
                try:
                    lm = extractor.extract_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), timestamp_ms=1)
                    print(f"    pose={lm.pose is not None} left_hand={lm.left_hand is not None} "
                          f"right_hand={lm.right_hand is not None}")
                    if lm.is_empty():
                        print("    nothing detected in that frame - make sure a person is in view; "
                              "this is a detection issue, not a camera issue")
                finally:
                    extractor.close()
        except Exception as exc:
            print(f"    extractor check failed: {type(exc).__name__}: {exc}")

    print("-" * 72)
    for hint in rep.diagnosis():
        print(f" * {hint}")

    if not rep.usable:
        print()
        print(camera_utils.no_camera_help())

    if rep.usable:
        return 0
    return 2 if not rep.gui_available else 1


if __name__ == "__main__":
    raise SystemExit(main())
