"""
Text-to-speech with graceful degradation.

The application must never crash because audio is unavailable (headless servers,
containers without sound devices, missing engine).  Therefore:

* ``gtts`` is used when installed and the machine has network access - it writes
  an MP3 and playback is attempted with whatever player exists;
* ``pyttsx3`` is used if present (offline, uses espeak/nsss);
* otherwise the speaker reports ``available == False`` and ``say()`` returns
  silently, and the UI shows "speech unavailable".

Speech is always triggered through :class:`backend.inference.predictor.PredictionEngine`,
which enforces "confident + stable + new-or-cooldown-expired" before calling here.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import config

LOG = logging.getLogger("tts")

#: players tried in order; the first one present on PATH is used
_PLAYERS = (
    ("ffplay", ["-nodisp", "-autoexit", "-loglevel", "quiet"]),
    ("mpg123", ["-q"]),
    ("afplay", []),
    ("aplay", ["-q"]),
    ("paplay", []),
    ("cvlc", ["--play-and-exit", "--intf", "dummy"]),
)


class Speaker:
    """Threaded, failure-tolerant text-to-speech."""

    def __init__(self, engine: Optional[str] = None, language: Optional[str] = None,
                 rate: Optional[int] = None, out_dir: Optional[Path] = None):
        self.engine = (engine or config.TTS_ENGINE).lower()
        self.language = language or config.TTS_LANGUAGE
        self.rate = int(rate or config.TTS_RATE)
        self.out_dir = Path(out_dir or (config.LOGS_DIR / "tts"))
        self.backend = "none"
        self.unavailable_reason = ""
        self._lock = threading.Lock()
        self._pyttsx = None
        self._detect()

    # ------------------------------------------------------------------
    def _detect(self) -> None:
        want = self.engine
        if want in ("auto", "gtts"):
            try:
                import gtts  # noqa: F401

                self._gtts = gtts
                self.backend = "gtts"
                self.player = self._find_player()
                if self.player is None:
                    self.unavailable_reason = (
                        "gTTS can synthesise audio but no command-line player "
                        "(ffplay/mpg123/afplay/aplay/paplay/cvlc) was found; "
                        "the MP3 is still written to logs/tts/."
                    )
                return
            except Exception as exc:  # pragma: no cover
                self.unavailable_reason = f"gtts import failed: {exc}"
        if want in ("auto", "pyttsx3"):
            try:
                import pyttsx3

                self._pyttsx = pyttsx3.init()
                self._pyttsx.setProperty("rate", self.rate)
                self.backend = "pyttsx3"
                return
            except Exception as exc:  # pragma: no cover
                self.unavailable_reason = f"pyttsx3 unavailable: {exc}"
        self.backend = "none"

    @staticmethod
    def _find_player():
        for name, args in _PLAYERS:
            if shutil.which(name):
                return (name, args)
        return None

    # ------------------------------------------------------------------
    @property
    def available(self) -> bool:
        return self.backend in ("gtts", "pyttsx3")

    def status(self) -> dict:
        return {"engine_requested": self.engine, "engine_active": self.backend,
                "available": self.available, "language": self.language,
                "reason": self.unavailable_reason or None}

    # ------------------------------------------------------------------
    def say(self, text: str, blocking: bool = False) -> Optional[Path]:
        """Speak ``text``. Returns the audio path when a file was produced."""
        text = (text or "").strip()
        if not text:
            return None
        if blocking:
            return self._say(text)
        threading.Thread(target=self._say, args=(text,), daemon=True).start()
        return None

    def _say(self, text: str) -> Optional[Path]:
        with self._lock:
            try:
                if self.backend == "pyttsx3" and self._pyttsx is not None:
                    self._pyttsx.say(text)
                    self._pyttsx.runAndWait()
                    return None
                if self.backend == "gtts":
                    return self._say_gtts(text)
            except Exception as exc:  # pragma: no cover - never propagate
                LOG.warning("speech failed: %s", exc)
            return None

    def _say_gtts(self, text: str) -> Optional[Path]:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        path = self.out_dir / f"speech-{stamp}-{abs(hash(text)) % 10000:04d}.mp3"
        self._gtts.gTTS(text=text, lang=self.language).save(str(path))
        player = getattr(self, "player", None)
        if player:
            name, args = player
            try:
                subprocess.run([name, *args, str(path)], check=False,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
            except Exception as exc:  # pragma: no cover
                LOG.warning("playback failed (%s): %s", name, exc)
        return path


def available_engines() -> dict:
    """Report which TTS backends this machine supports (used by /model-info)."""
    info = {"gtts": False, "pyttsx3": False, "player": None}
    try:
        import gtts  # noqa: F401

        info["gtts"] = True
    except Exception:
        pass
    try:
        import pyttsx3  # noqa: F401

        info["pyttsx3"] = True
    except Exception:
        pass
    info["player"] = Speaker._find_player()[0] if Speaker._find_player() else None
    info["platform"] = sys.platform
    return info
