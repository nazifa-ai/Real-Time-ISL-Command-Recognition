"""
Structured audit logging, human-feedback handling and error review.

Two append-only JSONL streams keep the design simple and honest:

``logs/predictions.jsonl``
    one record per emission (see the schema below);
``logs/feedback/feedback.jsonl``
    one record per human ACCEPT / CORRECT / REJECT, linked by ``event_id``.

Nothing in this project retrains from feedback automatically: corrections are
recorded for review, and only an explicit, opt-in personalisation run can use
them (see ``backend/personalization.py``).

Record schema (matches the project specification)
-------------------------------------------------
{"event_id", "timestamp", "model", "predicted_class", "confidence", "status",
 "latency_ms", "fps", "human_action", "corrected_class", "source", "profile"}
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import config

_LOCK = threading.Lock()
VALID_ACTIONS = ("ACCEPT", "CORRECT", "REJECT")


# --------------------------------------------------------------------------
# paths
# --------------------------------------------------------------------------
def predictions_path() -> Path:
    p = Path(config.LOG_PATHS["predictions"])
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def feedback_path() -> Path:
    p = Path(config.LOG_PATHS["feedback"])
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def failures_path() -> Path:
    p = Path(config.LOG_PATHS["audit"])
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _append(path: Path, record: Dict) -> None:
    with _LOCK:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")


def _iter_jsonl(path: Path) -> Iterable[Dict]:
    if not Path(path).exists():
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


# --------------------------------------------------------------------------
# writes
# --------------------------------------------------------------------------
def log_prediction(predicted_class: str, confidence: float, status: str,
                   latency_ms: float, model: str, fps: Optional[float] = None,
                   source: str = "api", profile: Optional[str] = None,
                   caption: Optional[str] = None, extra: Optional[Dict] = None) -> Dict:
    """Append one prediction event and return the stored record."""
    record = {
        "event_id": uuid.uuid4().hex[:12],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "predicted_class": predicted_class,
        "confidence": round(float(confidence), 6),
        "status": status,
        "latency_ms": round(float(latency_ms), 3),
        "fps": round(float(fps), 2) if fps is not None else None,
        "human_action": None,
        "corrected_class": None,
        "source": source,
        "profile": profile,
        "caption": caption,
    }
    if extra:
        record.update(extra)
    _append(predictions_path(), record)
    return record


def log_feedback(event_id: Optional[str], action: str, predicted_class: str,
                 corrected_class: Optional[str] = None, confidence: Optional[float] = None,
                 model: Optional[str] = None, note: Optional[str] = None,
                 profile: Optional[str] = None) -> Dict:
    """Record a human verdict. ``action`` must be ACCEPT / CORRECT / REJECT."""
    action = (action or "").strip().upper()
    if action not in VALID_ACTIONS:
        raise ValueError(f"action must be one of {VALID_ACTIONS}, got {action!r}")
    if action == "CORRECT" and not (corrected_class or "").strip():
        raise ValueError("CORRECT requires corrected_class")

    record = {
        "feedback_id": uuid.uuid4().hex[:12],
        "event_id": event_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "human_action": action,
        "predicted_class": predicted_class,
        "corrected_class": (corrected_class or None),
        "confidence": round(float(confidence), 6) if confidence is not None else None,
        "model": model,
        "note": note,
        "profile": profile,
    }
    _append(feedback_path(), record)
    return record


def log_failure(stage: str, message: str, model: Optional[str] = None,
                source: str = "api", session_id: Optional[str] = None) -> Dict:
    """Record a real pipeline failure without inventing a prediction event."""
    record = {
        "failure_id": uuid.uuid4().hex[:12],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "stage": str(stage)[:64], "message": str(message)[:1000],
        "model": model, "source": source, "session_id": session_id,
    }
    _append(failures_path(), record)
    return record


def read_failures(limit: int = 100) -> List[Dict]:
    return list(_iter_jsonl(failures_path()))[-max(1, min(int(limit), 1000)):][::-1]


# --------------------------------------------------------------------------
# reads (error review)
# --------------------------------------------------------------------------
def read_events(limit: int = 500, min_confidence: Optional[float] = None,
                max_confidence: Optional[float] = None, status: Optional[str] = None,
                predicted_class: Optional[str] = None, action: Optional[str] = None,
                model: Optional[str] = None, only_corrected: bool = False,
                only_rejected: bool = False, only_uncertain: bool = False,
                source: Optional[str] = None) -> List[Dict]:
    """Join predictions with feedback and apply the error-review filters."""
    predictions = list(_iter_jsonl(predictions_path()))
    feedback = {f.get("event_id"): f for f in _iter_jsonl(feedback_path()) if f.get("event_id")}

    merged: List[Dict] = []
    for p in predictions:
        fb = feedback.get(p.get("event_id"))
        row = dict(p)
        row["human_action"] = fb.get("human_action") if fb else p.get("human_action")
        row["corrected_class"] = fb.get("corrected_class") if fb else p.get("corrected_class")
        row["wrong"] = bool(row.get("corrected_class")) or row.get("human_action") == "REJECT"
        merged.append(row)

    def keep(r: Dict) -> bool:
        if min_confidence is not None and float(r.get("confidence") or 0) < float(min_confidence):
            return False
        if max_confidence is not None and float(r.get("confidence") or 0) > float(max_confidence):
            return False
        if status and (r.get("status") or "").upper() != status.upper():
            return False
        if predicted_class and (r.get("predicted_class") or "").upper() != predicted_class.upper():
            return False
        if action and (r.get("human_action") or "").upper() != action.upper():
            return False
        if model and r.get("model") != model:
            return False
        if source and r.get("source") != source:
            return False
        if only_corrected and not r.get("corrected_class"):
            return False
        if only_rejected and r.get("human_action") != "REJECT":
            return False
        if only_uncertain and (r.get("status") or "").upper() != config.UNCERTAIN_LABEL:
            return False
        return True

    filtered = [r for r in merged if keep(r)]
    filtered.sort(key=lambda r: r.get("timestamp") or "", reverse=True)
    return filtered[:limit]


def clear_logs(kind: str = "all") -> None:
    """Remove log files (used by tests; never called automatically)."""
    targets = []
    if kind in ("all", "predictions"):
        targets.append(predictions_path())
    if kind in ("all", "feedback"):
        targets.append(feedback_path())
    for t in targets:
        if t.exists():
            t.unlink()


# --------------------------------------------------------------------------
# monitoring (/metrics)
# --------------------------------------------------------------------------
def metrics_summary() -> Dict[str, object]:
    """Aggregate counters used by GET /metrics and the browser dashboard."""
    predictions = list(_iter_jsonl(predictions_path()))
    feedback = list(_iter_jsonl(feedback_path()))

    n = len(predictions)
    latencies = [float(p.get("latency_ms") or 0.0) for p in predictions if p.get("latency_ms")]
    confidences = [float(p.get("confidence") or 0.0) for p in predictions if p.get("confidence") is not None]
    status_counts = Counter((p.get("status") or "UNKNOWN") for p in predictions)
    per_class = Counter((p.get("predicted_class") or "UNKNOWN") for p in predictions)
    per_model = Counter((p.get("model") or "unknown") for p in predictions)
    actions = Counter((f.get("human_action") or "UNKNOWN") for f in feedback)

    corrected = [f for f in feedback if f.get("corrected_class")]
    acc_proxy = None
    if feedback:
        decided = [f for f in feedback if f.get("human_action") in ("ACCEPT", "CORRECT", "REJECT")]
        if decided:
            acc_proxy = sum(1 for f in decided if f.get("human_action") == "ACCEPT") / len(decided)

    def pct(values, q):
        if not values:
            return 0.0
        import numpy as np

        return float(np.percentile(values, q))

    return {
        "predictions_total": n,
        "feedback_total": len(feedback),
        "status_counts": dict(status_counts),
        "uncertain_rate": (status_counts.get(config.UNCERTAIN_LABEL, 0) / n) if n else 0.0,
        "confidence": {
            "mean": float(sum(confidences) / len(confidences)) if confidences else 0.0,
            "min": float(min(confidences)) if confidences else 0.0,
            "max": float(max(confidences)) if confidences else 0.0,
        },
        "latency_ms": {
            "mean": float(sum(latencies) / len(latencies)) if latencies else 0.0,
            "p50": pct(latencies, 50),
            "p95": pct(latencies, 95),
            "max": float(max(latencies)) if latencies else 0.0,
        },
        "throughput_fps": (1000.0 / (sum(latencies) / len(latencies))) if latencies else 0.0,
        "human_actions": dict(actions),
        "corrections_total": len(corrected),
        "human_agreement_rate": acc_proxy,
        "top_predicted_classes": dict(per_class.most_common(15)),
        "per_model": dict(per_model),
    }


def error_pairs(min_count: int = 1) -> List[Dict[str, object]]:
    """Most frequent prediction→correction confusions (for the error review view)."""
    pairs = Counter()
    for f in _iter_jsonl(feedback_path()):
        if f.get("corrected_class"):
            pairs[(f.get("predicted_class"), f.get("corrected_class"))] += 1
    return [{"predicted": p, "corrected": c, "count": n}
            for (p, c), n in pairs.most_common() if n >= min_count]
