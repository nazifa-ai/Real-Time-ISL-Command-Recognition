#!/usr/bin/env python3
"""
PHASE 12/13/14 -- real-time webcam recognition.

    camera -> MediaPipe -> temporal buffer -> model -> softmax
           -> confidence gate -> temporal smoothing -> stable prediction
           -> caption -> optional speech -> JSONL audit log

Keyboard controls (window must have focus)
------------------------------------------
    q / ESC  quit
    s        toggle speech on/off
    a        ACCEPT current prediction
    c        CORRECT current prediction (prompts in the terminal)
    r        REJECT current prediction
    1/2/3    switch model (lstm / gru / gru_mha)
    x        clear the current temporal window
    b        backspace: remove the last caption token

Headless machines (servers, Docker) have no camera: the script detects that and
prints the alternatives instead of failing obscurely.

Usage
-----
    python backend/inference/realtime.py
    python backend/inference/realtime.py --model gru --threshold 0.75 --no-speech
    python backend/inference/realtime.py --camera 1 --record out.mp4
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import config  # noqa: E402
from backend.inference import audit, camera as camera_utils  # noqa: E402
from backend.inference.frames import FrameBuffer  # noqa: E402
from backend.inference.predictor import PredictionEngine  # noqa: E402

LOG = logging.getLogger("realtime")


def _overlay(frame, result, engine, extra: str = "") -> np.ndarray:
    """Draw the HUD: caption, sign, confidence bar, status, fps, model."""
    import cv2

    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 96), (20, 20, 20), -1)
    cv2.putText(frame, f"Sign: {result.sign}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2)
    colour = (0, 200, 0) if result.status == "CONFIDENT" else (
        (0, 165, 255) if result.status == "UNCERTAIN" else (200, 200, 200))
    cv2.putText(frame, f"Status: {result.status}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, colour, 2)
    cv2.putText(frame, f"{result.confidence*100:5.1f}% ({result.model})", (10, 88),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    bar_x, bar_y, bar_w = w - 260, 20, 240
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 16), (80, 80, 80), -1)
    cv2.rectangle(frame, (bar_x, bar_y),
                  (bar_x + int(bar_w * min(1.0, result.confidence)), bar_y + 16), colour, -1)

    caption = engine.caption[-12:]
    cv2.rectangle(frame, (0, h - 46), (w, h), (20, 20, 20), -1)
    cv2.putText(frame, "Caption: " + " ".join(caption), (10, h - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 255, 120), 2)
    if extra:
        cv2.putText(frame, extra, (bar_x, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 220, 255), 1)
    return frame


def run_camera(engine: PredictionEngine, camera: int = config.CAMERA_INDEX,
               width: int = config.FRAME_WIDTH, height: int = config.FRAME_HEIGHT,
               speak: bool = True, log: bool = True, show: bool = True,
               record: Optional[str] = None, max_seconds: Optional[float] = None,
               profile: Optional[str] = None) -> dict:
    import cv2

    # A preview window needs a display, and cv2.imshow with no DISPLAY aborts the
    # process (Qt/xcb).  Probe once, and fall back to a windowless session.
    gui_ok, gui_note = camera_utils.gui_available()
    if show and not gui_ok:
        print(f"[info] no usable display ({gui_note}) - continuing without the preview window.\n"
              "       Run with --record out.mp4 to save an annotated video instead.")
        show = False

    cap, used_index, message = camera_utils.open_camera(preferred_index=camera, width=width, height=height)
    if cap is None:
        print(camera_utils.no_camera_help())
        return {"error": "camera_unavailable", "detail": message, "devices": camera_utils.device_nodes()}
    print(f"[info] {message}")
    if show:
        print("[info] preview window active - press q or ESC in the window to quit")
    engine.set_speech(speak)

    buffer = FrameBuffer()
    writer = None
    if record:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(record, fourcc, config.TARGET_FPS, (width, height))

    last = None
    fps_ema = 0.0
    t_start = time.time()
    emitted = 0
    print("Controls: q quit | s speech | a accept | c correct | r reject | 1/2/3 model | x clear window | b backspace")
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                time.sleep(0.02)
                continue
            t0 = time.time()
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            feats = buffer.push(rgb, timestamp_ms=int((time.time() - t_start) * 1000) + 1)

            if feats is not None:
                inst_fps = 1.0 / max(1e-3, time.time() - t0)
                fps_ema = inst_fps if fps_ema == 0 else 0.9 * fps_ema + 0.1 * inst_fps
                result = engine.predict(feats, speak=speak, fps=fps_ema)
                last = result
                if log and result.should_speak:
                    audit.log_prediction(result.sign, result.confidence, result.status,
                                         result.latency_ms, result.model, fps=fps_ema,
                                         source="realtime", profile=profile,
                                         caption=result.caption)
                    emitted += 1

            if last is not None:
                info = buffer.stats.as_dict()
                frame_bgr = _overlay(frame_bgr, last, engine,
                                     extra=f"fps {info['fps']:.1f} | hands L{info['left_hand_rate']*100:.0f}% "
                                           f"R{info['right_hand_rate']*100:.0f}%")
            elif show:
                cv2.putText(frame_bgr, "collecting frames...", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)

            if writer is not None:
                writer.write(frame_bgr)
            if show:
                cv2.imshow("ISL Recognition", frame_bgr)

            key = cv2.waitKey(1) & 0xFF if show else 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                engine.set_speech(not engine.speech_enabled)
                print(f"speech {'ON' if engine.speech_enabled else 'OFF'}")
            elif key in (ord("a"), ord("c"), ord("r")) and last is not None:
                _handle_action(key, last, engine, profile=profile)
            elif key in (ord("1"), ord("2"), ord("3")):
                name = {"1": "lstm", "2": "gru", "3": "gru_mha"}[chr(key)]
                try:
                    engine.switch_model(name)
                    print(f"model -> {name}")
                except FileNotFoundError as exc:
                    print(f"cannot switch: {exc}")
            elif key == ord("x"):
                engine.clear_window()
                buffer.reset()
            elif key == ord("b"):
                removed = engine.pop_last()
                if removed:
                    print(f"removed '{removed}' from caption")

            if max_seconds and (time.time() - t_start) > max_seconds:
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if show:
            cv2.destroyAllWindows()
        buffer.close()

    summary = {"emitted_predictions": emitted, "caption": " ".join(engine.caption),
               "seconds": round(time.time() - t_start, 1), **engine.stats(),
               "buffer": buffer.stats.as_dict()}
    print("\nSession summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    return summary


def _handle_action(key: int, result, engine: PredictionEngine, profile: Optional[str] = None) -> None:
    """Human override: ACCEPT / CORRECT / REJECT -> audit log."""
    action = {ord("a"): "ACCEPT", ord("c"): "CORRECT", ord("r"): "REJECT"}[key]
    corrected = None
    if action == "CORRECT":
        try:
            corrected = input(f"correct label for '{result.sign}' (e.g. WATER): ").strip().upper()
        except EOFError:
            return
        if not corrected:
            return
    taken = engine.pop_last() if action in ("CORRECT", "REJECT") else None
    record = audit.log_feedback(None, action, result.sign, corrected,
                                confidence=result.confidence, model=result.model, profile=profile)
    print(f"{action}: predicted={result.sign} corrected={corrected} (feedback {record['feedback_id']})"
          + (f" | caption token '{taken}' removed" if taken else ""))


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=config.DEFAULT_MODEL, choices=list(config.AVAILABLE_MODELS))
    ap.add_argument("--threshold", type=float, default=config.CONFIDENCE_THRESHOLD)
    ap.add_argument("--camera", type=int, default=config.CAMERA_INDEX)
    ap.add_argument("--list-cameras", action="store_true",
                    help="report devices, tested indices and display/GUI support, then exit")
    ap.add_argument("--width", type=int, default=config.FRAME_WIDTH)
    ap.add_argument("--height", type=int, default=config.FRAME_HEIGHT)
    ap.add_argument("--no-speech", action="store_true")
    ap.add_argument("--no-window", action="store_true", help="run without a GUI window (still logs)")
    ap.add_argument("--no-log", action="store_true")
    ap.add_argument("--record", default=None, help="write an annotated mp4 to this path")
    ap.add_argument("--seconds", type=float, default=None, help="stop after N seconds")
    ap.add_argument("--profile", default=None, help="personalisation profile name")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.list_cameras:
        rep = camera_utils.report()
        print(f"platform     : {rep.platform}")
        print(f"device nodes : {rep.device_nodes or 'none'}")
        print(f"DISPLAY      : {rep.display}")
        print(f"opencv GUI   : {rep.opencv_gui_backend} (usable: {rep.gui_available})")
        for probe in rep.probes:
            print(f"  index {probe.index}: opened={probe.opened} readable={probe.readable} "
                  f"shape={probe.shape} fps={probe.fps:.1f} {probe.error}")
        for hint in rep.diagnosis():
            print(f" * {hint}")
        return 0 if rep.usable else 1

    try:
        engine = PredictionEngine(model_name=args.model, threshold=args.threshold)
    except FileNotFoundError as exc:
        print(f"error: {exc}")
        return 2
    run_camera(engine, camera=args.camera, width=args.width, height=args.height,
               speak=not args.no_speech, log=not args.no_log, show=not args.no_window,
               record=args.record, max_seconds=args.seconds, profile=args.profile)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
