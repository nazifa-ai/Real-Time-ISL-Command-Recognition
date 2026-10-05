"""
Camera discovery and GUI capability detection.

A webcam needs **two** things that a server, a container, WSL, or an SSH session
usually does not have:

1. a capture device (``/dev/video*`` on Linux, an AVFoundation device on macOS,
   a DirectShow device on Windows);
2. a display, if you want the OpenCV preview window.

Missing either one is normal outside a desktop session, so every entry point in this
project asks this module first and degrades to a documented fallback instead of
crashing.  ``cv2.imshow`` in particular is *not* safe to call blind: with the Qt
backend and no ``DISPLAY`` the process is aborted by the C++ layer (SIGABRT), which
is why GUI support is probed in a **subprocess**.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import config


@dataclass
class CameraProbe:
    index: int
    opened: bool = False
    readable: bool = False
    shape: Tuple[int, int, int] | None = None
    fps: float = 0.0
    error: str = ""

    def as_dict(self) -> Dict[str, object]:
        return {
            "index": self.index, "opened": self.opened, "readable": self.readable,
            "shape": list(self.shape) if self.shape else None,
            "fps": round(self.fps, 2), "error": self.error,
        }


@dataclass
class CameraReport:
    platform: str
    device_nodes: List[str] = field(default_factory=list)
    display: Optional[str] = None
    gui_available: bool = False
    gui_note: str = ""
    probes: List[CameraProbe] = field(default_factory=list)
    opencv_gui_backend: str = ""

    @property
    def usable(self) -> bool:
        return any(p.readable for p in self.probes)

    @property
    def usable_index(self) -> Optional[int]:
        for p in self.probes:
            if p.readable:
                return p.index
        return None

    def diagnosis(self) -> List[str]:
        """Human-readable causes, most likely first, plus the matching fix."""
        hints: List[str] = []
        if not self.device_nodes and self.platform.lower().startswith("linux"):
            hints.append(
                "No /dev/video* device exists: this machine has no camera attached (typical for "
                "servers, containers, WSL, CI and cloud sandboxes)."
            )
        if not self.gui_available:
            hints.append(
                "No usable display (" + (self.gui_note or "DISPLAY unset") + "): OpenCV cannot open a "
                "preview window, and calling cv2.imshow here aborts the process. Use --no-window or a "
                "local desktop session."
            )
        if self.usable:
            hints.append(f"Camera index {self.usable_index} works - real-time mode can run here.")
        else:
            hints.append(
                "No camera could be opened on indices 0-3. If a camera IS attached, the usual causes "
                "are: (a) it is in use by another application, (b) OS camera permission was not granted "
                "to your terminal/browser, (c) it is not index 0 - try --camera 1 or 2."
            )
        return hints


def device_nodes() -> List[str]:
    if platform.system() == "Linux":
        return sorted(str(p) for p in Path("/dev").glob("video*"))
    return []


def gui_available(timeout: float = 15.0) -> Tuple[bool, str]:
    """Probe OpenCV HighGUI in a subprocess (a missing display can abort the parent)."""
    code = (
        "import cv2;"
        "cv2.namedWindow('isl_probe');cv2.destroyAllWindows();print('GUI_OK')"
    )
    env = dict(os.environ)
    if not env.get("DISPLAY") and platform.system() == "Linux":
        env["QT_QPA_PLATFORM"] = "offscreen"
    try:
        res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                             timeout=timeout, env=env)
    except Exception as exc:                                  # pragma: no cover
        return False, f"probe failed: {type(exc).__name__}: {exc}"
    if res.returncode == 0 and "GUI_OK" in res.stdout:
        return True, "opencv HighGUI available"
    tail = (res.stderr or res.stdout or "").strip().splitlines()
    return False, (tail[-1][:180] if tail else f"exit code {res.returncode}")


def probe_camera(index: int, warmup_frames: int = 2) -> CameraProbe:
    import cv2

    probe = CameraProbe(index=index)
    cap = None
    try:
        cap = cv2.VideoCapture(index)
        probe.opened = bool(cap.isOpened())
        if not probe.opened:
            probe.error = "device could not be opened (absent, busy, or wrong index)"
            return probe
        frame = None
        for _ in range(max(1, warmup_frames)):
            ok, frame = cap.read()
            if not ok:
                frame = None
                break
        if frame is None:
            probe.error = "opened but no frame could be read (driver/permission or capture in use)"
            return probe
        probe.readable = True
        probe.shape = tuple(int(x) for x in frame.shape)     # type: ignore[assignment]
        fps = cap.get(cv2.CAP_PROP_FPS)
        probe.fps = float(fps) if fps and fps > 0 else 0.0
        return probe
    except Exception as exc:                                  # pragma: no cover
        probe.error = f"{type(exc).__name__}: {exc}"
        return probe
    finally:
        if cap is not None:
            cap.release()


def probe_cameras(indices: Optional[List[int]] = None) -> List[CameraProbe]:
    if indices is None:
        indices = list(range(int(config.CAMERA_PROBE_MAX_INDEX) + 1))
    return [probe_camera(i) for i in indices]


def report(indices: Optional[List[int]] = None, check_gui: bool = True) -> CameraReport:
    import cv2

    rep = CameraReport(platform=f"{platform.system()} {platform.release()}",
                       device_nodes=device_nodes(), display=os.environ.get("DISPLAY"))
    rep.opencv_gui_backend = "Qt5" if "QT5" in cv2.getBuildInformation() else (
        "GTK" if "GTK" in cv2.getBuildInformation() else "none/headless build")
    if check_gui:
        rep.gui_available, rep.gui_note = gui_available()
    rep.probes = probe_cameras(indices)
    return rep


def open_camera(preferred_index: Optional[int] = None, width: Optional[int] = None,
                height: Optional[int] = None, fallback_indices: int = 3):
    """Open a working camera, searching a few indices if the preferred one fails.

    Returns ``(capture, index, message)``; ``capture`` is ``None`` when nothing works.
    """
    import cv2

    order = [config.CAMERA_INDEX if preferred_index is None else int(preferred_index)]
    order += [i for i in range(fallback_indices + 1) if i not in order]
    for idx in order:
        cap = cv2.VideoCapture(idx)
        ok = cap.isOpened()
        frame = None
        if ok:
            for _ in range(2):
                ok, frame = cap.read()
                if not ok:
                    break
        if ok and frame is not None:
            if width:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            if height:
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            msg = f"camera {idx} opened ({frame.shape[1]}x{frame.shape[0]})"
            if idx != order[0]:
                msg += f" - index {order[0]} was unavailable, fell back to {idx}"
            return cap, idx, msg
        cap.release()
    return None, None, (
        f"no camera could be opened on indices {order} "
        f"(devices found: {device_nodes() or 'none'})"
    )


def no_camera_help() -> str:
    """The message shown when capture is impossible on this machine."""
    return (
        "No working camera on this machine - this is expected for servers, containers, "
        "WSL/SSH sessions and cloud sandboxes, and it is not a bug in the model.\n"
        "  Diagnose : python run.py camera\n"
        "  Options  :\n"
        "    * python run.py batch --from-dataset --split test --limit 20   (dataset clips, no camera)\n"
        "    * python run.py batch --video path/to/clip.mp4                 (video file)\n"
        "    * open the browser dashboard with python run.py api                 (upload / dataset demos)\n"
        "    * on a desktop machine with a camera: python run.py realtime   (needs a display; add\n"
        "      --no-window if you only want terminal output + logs)\n"
        "    * inside Docker: pass the device through (--device /dev/video0) and run with --no-window,\n"
        "      or run realtime.py on the host instead (see docs/setup.md §6)."
    )
