"""Camera/GUI detection: the checks must be safe on a machine with no camera or no display."""

from __future__ import annotations

import pytest

import config
from backend.inference import camera as camera_utils


def test_device_nodes_returns_a_list():
    nodes = camera_utils.device_nodes()
    assert isinstance(nodes, list)
    assert all(n.startswith("/dev/video") for n in nodes)


def test_gui_probe_never_crashes_the_test_process():
    """cv2.imshow aborts the process without a display, so the probe must be isolated."""
    ok, note = camera_utils.gui_available()
    assert isinstance(ok, bool)
    assert isinstance(note, str) and note


def test_probe_of_a_missing_index_is_reported_not_raised():
    probe = camera_utils.probe_camera(index=99)
    assert probe.index == 99
    assert probe.readable is False
    if not probe.opened:
        assert "opened" in probe.error


def test_report_without_probing_every_index():
    rep = camera_utils.report(indices=[0], check_gui=True)
    assert rep.platform
    assert len(rep.probes) == 1
    assert isinstance(rep.usable, bool)
    assert isinstance(rep.gui_available, bool)


def test_diagnosis_always_explains_something():
    rep = camera_utils.report(indices=[0], check_gui=True)
    hints = rep.diagnosis()
    assert hints and all(isinstance(h, str) and h for h in hints)


def test_open_camera_returns_a_triple_and_never_raises():
    cap, index, message = camera_utils.open_camera()
    try:
        assert isinstance(message, str) and message
        if cap is not None:
            assert index is not None and index >= 0
        else:
            assert index is None
            assert "indices" in message
    finally:
        if cap is not None:
            cap.release()


def test_no_camera_help_lists_real_fallbacks():
    text = camera_utils.no_camera_help()
    for token in ("run.py camera", "batch --from-dataset", "batch --video", "browser dashboard", "realtime"):
        assert token in text, token


def test_probe_max_index_is_configurable():
    assert config.CAMERA_PROBE_MAX_INDEX >= 0


@pytest.mark.parametrize("index", [0])
def test_preferred_index_is_tried_first(index, monkeypatch):
    """open_camera must report which index produced frames."""
    cap, used, message = camera_utils.open_camera(preferred_index=index, fallback_indices=0)
    try:
        if cap is None:
            assert f"{index}" in message
        else:
            assert used == index
    finally:
        if cap is not None:
            cap.release()
