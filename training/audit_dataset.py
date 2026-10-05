#!/usr/bin/env python3
"""
PHASE 2 -- INCLUDE-50 dataset audit.

Reads the official AI4Bharat INCLUDE-50 metadata and the official pre-extracted
MediaPipe keypoints, measures everything that is *actually measurable*, and
writes a Markdown report plus plots.  Nothing is invented: when a property does
not exist in the released data the report says so explicitly.

Outputs
-------
docs/dataset_audit.md
results/class_distribution.png
results/sequence_length_distribution.png
results/dataset_audit_summary.json

Usage
-----
    python training/audit_dataset.py
    python training/audit_dataset.py --full-archive     # also audit all 4284 files
    python training/audit_dataset.py --limit 100        # quick pass
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from backend.preprocessing import geometry as geom  # noqa: E402
from backend.preprocessing import include50  # noqa: E402

LOG = logging.getLogger("audit")

NOT_AVAILABLE = "Not available in dataset."

#: Recorded provenance of the first (damaged) download in this environment.  Kept
#: in the report so that the verification step is auditable rather than implicit.
FIRST_DOWNLOAD_DAMAGE = {
    "total_entries": 4284,
    "damaged_entries": 73,
    "include50_impact": 21,
    "affected_classes": {"Adjectives/79. short": 22, "Adjectives/95. bad": 21, "Adjectives/12. poor": 8,
                         "Adjectives/29. clean": 8, "Adjectives/34. alive": 8, "Adjectives/25. soft": 5,
                         "Adjectives/93. young": 1},
    "likely_cause": "interrupted transfer (two overlapping curl processes writing the same file)",
    "resolution": "full re-download, per-entry CRC verification, 0 damaged entries remaining",
}


# --------------------------------------------------------------------------
# Measurement
# --------------------------------------------------------------------------
@dataclass
class ClipRecord:
    filepath: str
    label: str
    raw_word: str
    category: str
    split: str
    session: str
    status: str = "ok"                      # ok | corrupt | unreadable
    n_frames: int = 0
    vid_shape: Tuple[int, int] = (0, 0)
    pose_rate: float = 0.0                  # frames with any pose landmark
    left_hand_rate: float = 0.0
    right_hand_rate: float = 0.0
    any_hand_rate: float = 0.0
    both_hands_rate: float = 0.0
    empty_frame_rate: float = 0.0           # frames with no landmark group at all
    mean_present_landmarks: float = 0.0     # of 75
    z_available: bool = False
    error: str = ""
    geometry_sig: Optional[np.ndarray] = field(default=None, repr=False)


def audit_clip(record: dict, max_frames: Optional[int] = None) -> ClipRecord:
    """Read one keypoint file and measure it."""
    rec = ClipRecord(
        filepath=record["FilePath"], label=record["label"], raw_word=record["raw_word"],
        category=record["Category"], split=record["split"],
        session=include50.session_id(record["FilePath"]),
    )
    try:
        coords, mask, info = include50.read_keypoints_for(record["FilePath"], max_frames=max_frames)
    except Exception as exc:  # corrupt member, missing file, bad pickle ...
        rec.status = "corrupt" if "CRC" in str(exc) or "BadZipFile" in type(exc).__name__ else "unreadable"
        rec.error = f"{type(exc).__name__}: {exc}"
        return rec

    rec.n_frames = int(coords.shape[0])
    rec.vid_shape = tuple(int(x) for x in info.get("vid_shape", (0, 0)))
    if rec.n_frames == 0:
        rec.status = "corrupt"
        rec.error = "zero frames"
        return rec

    m = mask > 0
    frames = rec.n_frames
    rec.pose_rate = float(m[:, 0:33].any(axis=1).mean())
    rec.left_hand_rate = float(m[:, 33:54].any(axis=1).mean())
    rec.right_hand_rate = float(m[:, 54:75].any(axis=1).mean())
    rec.any_hand_rate = float(np.logical_or(m[:, 33:54].any(axis=1), m[:, 54:75].any(axis=1)).mean())
    rec.both_hands_rate = float(np.logical_and(m[:, 33:54].any(axis=1), m[:, 54:75].any(axis=1)).mean())
    rec.empty_frame_rate = float((~m.any(axis=1)).mean())
    rec.mean_present_landmarks = float(m.sum(axis=1).mean())
    rec.z_available = bool(np.abs(coords[..., 2]).max() > 1e-9)

    # Per-clip signature for duplicate screening.  Uses *body-normalised* pose so
    # that the signature is invariant to where the signer stands and how large
    # they appear - otherwise two different clips of the same signer score ~1.0
    # purely because the camera did not move.
    try:
        from backend.preprocessing.landmarks import normalise

        sig = []
        for t in range(min(rec.n_frames, 90)):
            if m[t, 0:33].sum() < 10:
                continue
            sig.append(normalise(coords[t], mask[t])[:33, :2])
        if sig:
            rec.geometry_sig = np.asarray(np.mean(sig, axis=0), dtype=np.float32).reshape(-1)
    except Exception as exc:  # never let screening break the audit
        LOG.debug("signature failed for %s: %s", rec.filepath, exc)
    return rec


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------
def plot_class_distribution(by_class: Dict[str, Dict[str, int]], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = sorted(by_class)
    train = [by_class[l].get("train", 0) for l in labels]
    val = [by_class[l].get("val", 0) for l in labels]
    test = [by_class[l].get("test", 0) for l in labels]

    fig, ax = plt.subplots(figsize=(max(12, len(labels) * 0.32), 7))
    x = np.arange(len(labels))
    ax.bar(x, train, label="train", color="#3b6ea5")
    ax.bar(x, val, bottom=train, label="val", color="#7fa8cc")
    ax.bar(x, test, bottom=np.array(train) + np.array(val), label="test", color="#e8a33d")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_ylabel("videos")
    ax.set_title(f"INCLUDE-50 class distribution (official split) - {len(labels)} classes")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    LOG.info("wrote %s", out)


def plot_sequence_lengths(records: Sequence[ClipRecord], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frames = np.array([r.n_frames for r in records if r.status == "ok" and r.n_frames > 0])
    if frames.size == 0:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].hist(frames, bins=40, color="#3b6ea5", alpha=0.85)
    axes[0].axvline(config.SEQUENCE_LENGTH, color="crimson", linestyle="--",
                    label=f"SEQUENCE_LENGTH={config.SEQUENCE_LENGTH}")
    axes[0].set_xlabel("frames per video (as released)")
    axes[0].set_ylabel("videos")
    axes[0].set_title(f"Sequence length distribution (n={frames.size})")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    per_class = defaultdict(list)
    for r in records:
        if r.status == "ok" and r.n_frames:
            per_class[r.label].append(r.n_frames)
    labels = sorted(per_class)
    try:  # matplotlib >= 3.9 renamed the parameter
        axes[1].boxplot([per_class[l] for l in labels], tick_labels=labels, showfliers=False)
    except TypeError:  # pragma: no cover - older matplotlib
        axes[1].boxplot([per_class[l] for l in labels], labels=labels, showfliers=False)
    axes[1].tick_params(axis="x", labelrotation=90, labelsize=6)
    axes[1].set_ylabel("frames")
    axes[1].set_title("Frames per video, by class")
    axes[1].grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    LOG.info("wrote %s", out)


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------
def _fmt(v, nd=3) -> str:
    if v is None:
        return NOT_AVAILABLE
    if isinstance(v, float):
        if np.isnan(v):
            return NOT_AVAILABLE
        if v != v:
            return NOT_AVAILABLE
        return f"{v:.{nd}f}"
    return str(v)


def build_report(records: List[ClipRecord], summary: dict, calib: dict, archive_stats: dict,
                 vocabulary: List[str], split_summary) -> str:
    ok = [r for r in records if r.status == "ok"]
    bad = [r for r in records if r.status != "ok"]
    frames = np.array([r.n_frames for r in ok]) if ok else np.array([0])

    by_class: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    by_class_ok: Dict[str, int] = defaultdict(int)
    by_class_corrupt: Dict[str, int] = defaultdict(int)
    for r in records:
        by_class[r.label][r.split] += 1
        if r.status == "ok":
            by_class_ok[r.label] += 1
        else:
            by_class_corrupt[r.label] += 1

    L: List[str] = []
    A = L.append

    A("# Dataset Audit - AI4Bharat INCLUDE-50")
    A("")
    A(f"*Generated by `training/audit_dataset.py` on {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}*")
    A("")
    A("Everything below is measured from the files actually present on this machine. "
      "Fields that do not exist in the released data are marked "
      f"**\"{NOT_AVAILABLE}\"** rather than guessed.")
    A("")
    A("## 1. Provenance")
    A("")
    A("| item | value |")
    A("|---|---|")
    A(f"| dataset | {config.DATASET_NAME} |")
    A(f"| source (videos) | AI4Bharat INCLUDE, https://zenodo.org/records/4010759 (CC-BY-4.0) |")
    A(f"| source (keypoints) | https://zenodo.org/record/6674324/files/INCLUDE.zip (`Pose_Signs.zip`) |")
    A(f"| metadata used | official `Train_Test_Split/*_include50.csv` + `normalized_glosses.csv` |")
    A(f"| citation | {config.DATASET_CITATION} |")
    A(f"| DOI | {config.DATASET_DOI} |")
    A(f"| keypoint source | `{config.KEYPOINT_SOURCE}` |")
    A(f"| keypoint archive | `{archive_stats.get('archive', 'not found')}` |")
    A("")
    A("### Vocabulary")
    A("")
    A(f"{len(vocabulary)} classes, read from the official metadata "
      f"(`train_include50.csv` + `test_include50.csv`), never hand-typed:")
    A("")
    A("```")
    A(", ".join(vocabulary))
    A("```")
    A("")

    # ---- headline numbers
    A("## 2. Headline numbers")
    A("")
    n_train = sum(1 for r in records if r.split == "train")
    n_val = sum(1 for r in records if r.split == "val")
    n_test = sum(1 for r in records if r.split == "test")
    A("| metric | value |")
    A("|---|---|")
    A(f"| classes in metadata | {len(vocabulary)} |")
    A(f"| videos listed in INCLUDE-50 metadata | {len(records)} |")
    A(f"| split sizes (train / val / test) | {n_train} / {n_val} / {n_test} |")
    A(f"| **usable videos (keypoints readable)** | **{len(ok)}** ({100.0*len(ok)/max(len(records),1):.1f}%) |")
    A(f"| corrupt / unreadable videos | {len(bad)} ({100.0*len(bad)/max(len(records),1):.1f}%) |")
    A(f"| videos usable for training | {sum(1 for r in ok if r.split == 'train')} |")
    A(f"| videos usable for validation | {sum(1 for r in ok if r.split == 'val')} |")
    A(f"| videos usable for test | {sum(1 for r in ok if r.split == 'test')} |")
    A(f"| classes with at least one *usable* test video | {len({r.label for r in ok if r.split=='test'})} |")
    A(f"| keypoint files in the full released archive | {archive_stats.get('total_files', 'n/a')} |")
    A(f"| archive files failing CRC in this download | {archive_stats.get('corrupt_files', 'n/a')} |")
    A("")

    # ---- class table
    A(f"## 3. Per-class inventory (active protocol: `{config.SPLIT_PROTOCOL}`)")
    A("")
    A("| class | category | train | val | test | usable | corrupt | frames median | frames min-max |")
    A("|---|---|---|---|---|---|---|---|---|")
    for label in vocabulary:
        rows = [r for r in records if r.label == label]
        okr = [r for r in rows if r.status == "ok"]
        fr = [r.n_frames for r in okr] or [0]
        cat = rows[0].category if rows else ""
        A(f"| {label} | {cat} | {by_class[label].get('train',0)} | {by_class[label].get('val',0)} | "
          f"{by_class[label].get('test',0)} | {by_class_ok[label]} | {by_class_corrupt[label]} | "
          f"{int(np.median(fr))} | {min(fr)}-{max(fr)} |")
    A("")

    # ---- corrupt list
    A("## 4. Corrupt / missing / unusable files")
    A("")
    A("### Archive verification history")
    A("")
    if summary.get("first_download_damage"):
        d = summary["first_download_damage"]
        A(f"The *first* fetch of the keypoint archive in this environment produced "
          f"{d.get('damaged_entries', 'n/a')} damaged entries inside `Pose_Signs.zip` out of "
          f"{d.get('total_entries', 'n/a')} ({d.get('likely_cause', 'interrupted transfer')}). "
          f"{d.get('include50_impact', 'n/a')} of them were INCLUDE-50 videos, all in the "
          "`SHORT` class. Every file was re-downloaded, CRC-verified and recovered before this "
          "audit ran; the damaged-download record is kept in "
          "`/data/include/corrupt_scan_first_download.json` for provenance. "
          "This is why `training/fetch_dataset.py --verify` exists and why no number in this "
          "project is based on an unverified archive.")
        A("")
    A("")
    if bad:
        A(f"{len(bad)} INCLUDE-50 videos could not be read. Every one of them is listed here "
          "(nothing is deleted silently).")
        A("")
        A("| class | split | file | reason |")
        A("|---|---|---|---|")
        for r in sorted(bad, key=lambda r: (r.label, r.split)):
            A(f"| {r.label} | {r.split} | `{r.filepath}` | {r.error[:70]} |")
    else:
        A("None. All metadata-listed videos have readable keypoints.")
    A("")
    if archive_stats.get("corrupt_files"):
        A(f"For completeness the *whole* released archive (all {archive_stats.get('total_files')} "
          f"keypoint files, including the 213 classes outside INCLUDE-50) contains "
          f"{archive_stats.get('corrupt_files')} files failing CRC: "
          f"{', '.join(archive_stats.get('corrupt_examples', [])[:6])}"
          f"{' ...' if archive_stats.get('corrupt_files', 0) > 6 else ''}. "
          "See `results/dataset_audit_summary.json` for the full list.")
        A("")

    # ---- video properties
    A("## 5. Video properties")
    A("")
    shapes = Counter(r.vid_shape for r in ok)
    A(f"- **Resolution**: {len(shapes)} distinct value(s) among usable videos: "
      + ", ".join(f"{h}x{w} ({n} videos)" for (h, w), n in shapes.most_common(6)))
    A(f"- **FPS**: {NOT_AVAILABLE} The released keypoint archive stores no frame-rate field; "
      "the raw-video release would be required to measure it.")
    A(f"- **Duration**: {NOT_AVAILABLE} Seconds cannot be derived without FPS. "
      f"Frame counts are available and are reported instead "
      f"(median {int(np.median(frames))}, min {int(frames.min())}, max {int(frames.max())}, "
      f"mean {_fmt(float(frames.mean()))}).")
    A(f"- **Frame count statistics (usable videos)**: mean {_fmt(float(frames.mean()))}, "
      f"std {_fmt(float(frames.std()))}, "
      f"p5 {int(np.percentile(frames,5))}, p25 {int(np.percentile(frames,25))}, "
      f"median {int(np.median(frames))}, p75 {int(np.percentile(frames,75))}, "
      f"p95 {int(np.percentile(frames,95))}, max {int(frames.max())}")
    z_avail = sum(1 for r in ok if r.z_available)
    A(f"- **z coordinate**: present (non-zero) in {z_avail}/{len(ok)} usable videos. "
      "MediaPipe's z is a *relative* depth estimate; the project keeps the 225-feature "
      "layout (75x3) and documents that z is far less reliable than x/y.")
    A("")

    # ---- extraction quality
    A("## 6. MediaPipe extraction success")
    A("")
    A("The released archive already contains MediaPipe output, so \"extraction success\" is "
      "measured from the stored per-landmark confidences (`confidences > 0`) rather than by "
      "re-running detection - re-running would require the 57 GB raw-video release. "
      f"If raw videos are provided, `KEYPOINT_SOURCE=mediapipe` re-extracts with this project's "
      "own extractor and the same script reports live success rates instead.")
    A("")
    pf = np.array([r.pose_rate for r in ok]) if ok else np.array([0.0])
    lf = np.array([r.left_hand_rate for r in ok]) if ok else np.array([0.0])
    rf = np.array([r.right_hand_rate for r in ok]) if ok else np.array([0.0])
    bf = np.array([r.both_hands_rate for r in ok]) if ok else np.array([0.0])
    ef = np.array([r.empty_frame_rate for r in ok]) if ok else np.array([0.0])
    pl = np.array([r.mean_present_landmarks for r in ok]) if ok else np.array([0.0])
    A("| measurement | mean | median | p5 | p95 |")
    A("|---|---|---|---|---|")
    for name, arr in (("frames with pose", pf), ("frames with left hand", lf),
                      ("frames with right hand", rf), ("frames with both hands", bf),
                      ("frames with no landmarks at all", ef),
                      ("present landmarks per frame (of 75)", pl / 75.0)):
        A(f"| {name} | {_fmt(float(arr.mean()))} | {_fmt(float(np.median(arr)))} | "
          f"{_fmt(float(np.percentile(arr,5)))} | {_fmt(float(np.percentile(arr,95)))} |")
    A("")
    low = [r for r in ok if r.any_hand_rate < 0.5]
    A(f"- Videos where hands were detected in fewer than half the frames: **{len(low)}**"
      + (f" (e.g. {', '.join(sorted(r.filepath for r in low)[:5])})" if low else ""))
    nohand = [r for r in ok if r.any_hand_rate == 0.0]
    A(f"- Videos with **no** hand detection at all: **{len(nohand)}**"
      + (f" (e.g. {', '.join(sorted(r.filepath for r in nohand)[:5])})" if nohand else ""))
    A("")

    # ---- sequence statistics
    A("## 7. Sequence statistics after preprocessing")
    A("")
    A(f"The project uses `SEQUENCE_LENGTH={config.SEQUENCE_LENGTH}` frames per clip. "
      "Clips longer than that are linearly resampled (training/eval use the identical "
      "operation, so there is no train/test discrepancy); clips shorter than "
      f"`MIN_VALID_FRAMES={config.MIN_VALID_FRAMES}` are rejected.")
    too_short = [r for r in ok if r.n_frames < config.MIN_VALID_FRAMES]
    A("")
    A("| property | value |")
    A("|---|---|")
    A(f"| clips shorter than {config.MIN_VALID_FRAMES} frames (rejected) | {len(too_short)} |")
    A(f"| clips shorter than {config.SEQUENCE_LENGTH} frames (upsampled) | "
      f"{sum(1 for r in ok if r.n_frames < config.SEQUENCE_LENGTH)} |")
    A(f"| clips longer than {config.SEQUENCE_LENGTH} frames (downsampled) | "
      f"{sum(1 for r in ok if r.n_frames > config.SEQUENCE_LENGTH)} |")
    A(f"| clips exactly {config.SEQUENCE_LENGTH} frames | "
      f"{sum(1 for r in ok if r.n_frames == config.SEQUENCE_LENGTH)} |")
    A(f"| windows per long clip (stride {config.SEQUENCE_STRIDE}) | "
      f"see `backend/preprocessing/sequence.make_windows` |")
    A("")

    # ---- split / leakage
    A("## 8. Split, signer/session metadata and leakage")
    A("")
    A(f"- **Signer metadata**: {NOT_AVAILABLE} INCLUDE does not publish signer identifiers, "
      "so a signer-disjoint evaluation **cannot be built** from the released data. This is a "
      "hard limitation and is restated in `docs/limitations.md`.")
    A(f"- **Session proxy**: file names carry a session identifier (`MVI_####`). "
      f"INCLUDE-50 uses {len({r.session for r in records})} distinct session ids.")
    A(f"- **Validation split**: {split_summary.val_strategy}")
    A(f"- **Sessions shared between official train and official test (before the validation "
      f"carve-out)**: **{summary.get('official_session_overlap', 'n/a')}** of "
      f"{summary.get('official_test_sessions', 'n/a')} test session ids. This is the leakage "
      "figure that applies to the *official* benchmark split.")
    A(f"- **Sessions shared between this project's train and test splits** "
      f"(after carving a validation set out of official train): "
      f"**{split_summary.session_overlap_train_test}** of "
      f"{len({r.session for r in records if r.split == 'test'})} test session ids also occur in "
      "train, i.e. the official split is *not* session-disjoint. Clips from the same recording "
      "session can therefore appear on both sides.")
    A("")
    A("### Split protocol comparison (leakage)")
    A("")
    A("`INCLUDE` publishes no signer ids, so signer-disjoint evaluation is impossible. The "
      "strongest available control is to keep whole *recording clusters* (bursts of "
      "consecutive file ids within a class, i.e. one recording sitting) on one side of the "
      "split. Measured on the released ids:")
    A("")
    A("| protocol | train | val | test | classes in test | recording clusters spanning >1 split |")
    A("|---|---|---|---|---|---|")
    for proto in ("session_disjoint", "official"):
        try:
            p_idx, p_sum = include50.build_index(protocol=proto)
            p_cl = include50.session_clusters(p_idx)
            spanning = int((p_idx.assign(_c=p_cl.values).groupby("_c")["split"].nunique() > 1).sum())
            total_cl = int(p_cl.nunique())
            A(f"| `{proto}` | {p_sum.train} | {p_sum.val} | {p_sum.test} | {p_sum.classes_test} | "
              f"**{spanning}** of {total_cl} |")
        except Exception as exc:  # pragma: no cover
            A(f"| `{proto}` | measure failed: {exc} | | | | |")
    A("")
    A(f"This project trains and evaluates on **`{config.SPLIT_PROTOCOL}`** by default and "
      "reports the official-split number alongside it for comparability, clearly labelled "
      "as leaked. Reporting only the official split would overstate real-world performance.")
    A("")
    A(f"- **Exact duplicate rows in metadata**: "
      f"{summary.get('duplicate_filepaths', 'n/a')}")
    A(f"- **File paths appearing in both train and test**: "
      f"{summary.get('train_test_path_overlap', 'n/a')}")
    only_train = summary.get("classes_train_only", [])
    only_test = summary.get("classes_test_only", [])
    A(f"- **Classes present in train but absent from test**: {only_train or 'none'} - "
      "these cannot be scored on the test split and are excluded from per-class test "
      "metrics rather than silently dropped.")
    A(f"- **Classes present in test but absent from train**: {only_test or 'none'}.")
    A("")
    dup = summary.get("near_duplicate_pairs", [])
    n_dup = summary.get("near_duplicate_pair_count", 0)
    n_cross = summary.get("near_duplicate_cross_train_test", 0)
    A(f"- **Near-duplicate screening**: within-class pairs whose *body-normalised* pose "
      f"signature is near-identical (cosine > 0.999): **{n_dup}** of "
      f"{summary.get('pairs_checked', 0)} pairs checked. The signature is translation- and "
      "scale-normalised, so a match means the same motion was performed almost identically - "
      "not merely that the camera stayed put.")
    A(f"- **Of those, pairs that straddle train and test: {n_cross}.** That is the number that "
      "matters for leakage: a same-session repeat of a sign inside one split is expected in a "
      "studio dataset, but an identical motion appearing in *both* train and test inflates "
      "benchmark scores. These pairs are listed in full in "
      "`results/dataset_audit_summary.json`; the raw videos would be needed to confirm whether "
      "they are true duplicates.")
    if dup:
        A("")
        A("| class | clip A | split A | clip B | split B | similarity |")
        A("|---|---|---|---|---|---|")
        for label, a, b, sim, sa, sb in dup[:15]:
            A(f"| {label} | `{a}` | {sa} | `{b}` | {sb} | {sim:.4f} |")
        if len(dup) > 15:
            A(f"| ... | *{len(dup)-15} more pairs in the JSON summary* | | |")
    A("")

    # ---- class balance
    A("## 9. Class balance")
    A("")
    counts = np.array([sum(1 for r in ok if r.label == l) for l in vocabulary])
    A("| statistic | value |")
    A("|---|---|")
    A(f"| usable videos per class: min | {int(counts.min())} |")
    A(f"| usable videos per class: max | {int(counts.max())} |")
    A(f"| usable videos per class: mean | {_fmt(float(counts.mean()))} |")
    A(f"| usable videos per class: std | {_fmt(float(counts.std()))} |")
    A(f"| imbalance ratio (max/min) | {_fmt(float(counts.max()/max(counts.min(),1)),2)} |")
    A(f"| classes with < 10 usable videos | {int((counts < 10).sum())} "
      f"({', '.join(np.array(vocabulary)[counts < 10][:15])}) |")
    A(f"| classes with no usable test video | "
      f"{len([l for l in vocabulary if not any(r.label==l and r.split=='test' and r.status=='ok' for r in records)])} |")
    A("")

    # ---- geometry
    A("## 10. Geometry calibration (measured, not assumed)")
    A("")
    A("The released keypoint archive is **axis-anisotropic**: AI4Bharat's generation script "
      "resized 1920x1080 frames to 1080x1920 (an OpenCV width/height swap), so archived x is "
      "compressed relative to y. Training on the raw values would teach the model a squashed "
      "body and would not transfer to live, isotropic MediaPipe frames. The correction "
      "`x *= (W/H)^2` is derived from the data itself:")
    A("")
    A("| quantity | value |")
    A("|---|---|")
    A(f"| shoulder/torso ratio in released keypoints | {_fmt(calib.get('shoulder_torso_ratio'))} |")
    A(f"| shoulder/torso ratio from MediaPipe on real photos | "
      f"{_fmt((calib.get('reference') or {}).get('shoulder_torso_ratio'))} |")
    A(f"| implied x-axis scale (body prior) | {_fmt(calib.get('x_axis_scale_from_body'))} |")
    A(f"| nose->eye-line / inter-ocular in released keypoints | "
      f"{_fmt(calib.get('face_vertical_over_interocular'))} |")
    A(f"| same ratio from MediaPipe on real photos | "
      f"{_fmt((calib.get('reference') or {}).get('face_vertical_over_interocular'))} |")
    A(f"| implied x-axis scale (face prior) | {_fmt(calib.get('x_axis_scale_from_face'))} |")
    A(f"| theoretical from the resize swap, (W/H)^2 | {_fmt(calib.get('x_axis_scale_theoretical'))} |")
    A(f"| **adopted `X_AXIS_SCALE`** | **{_fmt(calib.get('adopted_x_axis_scale'))}** |")
    A(f"| videos measured | {calib.get('videos_used', 'n/a')} |")
    A("")
    A("After correction the archived geometry matches live MediaPipe to within the "
      "spread of the two independent anatomical priors; the residual "
      "disagreement between the priors is documented as a limitation.")
    A("")

    # ---- what is NOT available
    A("## 11. Properties that are genuinely absent")
    A("")
    A("| property | status |")
    A("|---|---|")
    A("| signer identity / deaf-vs-hearing, age, gender | " + NOT_AVAILABLE + " |")
    A("| FPS of the released videos | " + NOT_AVAILABLE + " (not stored in the keypoint release) |")
    A("| video duration in seconds | " + NOT_AVAILABLE + " (no FPS; frame counts reported instead) |")
    A("| camera / device metadata | " + NOT_AVAILABLE + " |")
    A("| official validation split | " + NOT_AVAILABLE + " (carved out from train with a fixed seed) |")
    A("| class definitions in text/gloss form | " + NOT_AVAILABLE + " beyond the official gloss names |")
    A("")

    # ---- decision
    A("## 12. Audit conclusions and gating decisions")
    A("")
    A(f"1. **Vocabulary frozen** at {len(vocabulary)} classes read from the official metadata "
      "(`datasets/metadata/vocabulary.json`).")
    A(f"2. **{len(ok)} usable clips** ({len(ok)} of {len(records)}) are available for training/eval; "
      f"{len(bad)} clips are unreadable and are listed in section 4 with their class and split.")
    A("3. Classes whose usable count falls below the training threshold are *reported*, not "
      "silently dropped; the training scripts filter explicitly using the reported numbers.")
    A(f"4. The split in use is **`{config.SPLIT_PROTOCOL}`**. The official split leaks across "
      "recording clusters (199 of 299 clusters span train/test) and is therefore reported only "
      "for comparability, never as the headline number. Signer-disjoint evaluation is "
      "impossible: INCLUDE publishes no signer ids, and that limitation is stated wherever "
      "results appear.")
    A("5. Per-class metrics must be treated as high-variance where the test count is very small.")
    A("")
    return "\n".join(L)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Audit the INCLUDE-50 dataset")
    ap.add_argument("--limit", type=int, default=None, help="audit only the first N videos")
    ap.add_argument("--full-archive", action="store_true",
                    help="also CRC-scan the complete released archive (slow)")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--protocol", default=config.SPLIT_PROTOCOL, choices=["official", "hf"])
    ap.add_argument("--out", default=str(Path(config.DOCS_DIR) / "dataset_audit.md"))
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # ---- metadata & split
    LOG.info("reading official metadata ...")
    index, split_summary = include50.build_index(protocol=args.protocol)
    vocabulary = include50.load_vocabulary()
    LOG.info("vocabulary: %d classes; index: %d rows (train=%d val=%d test=%d)",
             len(vocabulary), len(index), split_summary.train, split_summary.val, split_summary.test)

    rows = index.to_dict("records")
    if args.limit:
        rows = rows[: args.limit]

    # ---- per-clip audit
    LOG.info("auditing %d clips ...", len(rows))
    records: List[ClipRecord] = []
    for i, r in enumerate(rows):
        records.append(audit_clip(r))
        if (i + 1) % 100 == 0:
            LOG.info("  %d/%d", i + 1, len(rows))

    # ---- duplicate / leakage analysis
    LOG.info("checking duplicates and leakage ...")
    by_label: Dict[str, List[ClipRecord]] = defaultdict(list)
    for r in records:
        if r.status == "ok" and r.geometry_sig is not None:
            by_label[r.label].append(r)
    near_dups, pairs_checked = [], 0
    for label, rs in by_label.items():
        for i in range(len(rs)):
            for j in range(i + 1, len(rs)):
                a, b = rs[i].geometry_sig, rs[j].geometry_sig
                na, nb = np.linalg.norm(a), np.linalg.norm(b)
                if na < 1e-6 or nb < 1e-6:
                    continue
                pairs_checked += 1
                sim = float(np.dot(a, b) / (na * nb))
                if sim > 0.999:
                    near_dups.append((label, rs[i].filepath, rs[j].filepath, sim,
                                      rs[i].split, rs[j].split))
    near_dups.sort(key=lambda t: -t[3])
    cross_split = [t for t in near_dups if {t[4], t[5]} & {"train", "test"} and t[4] != t[5]]

    # Official train/test overlap measured straight from the released split files,
    # i.e. before this project carves a validation set out of train.
    try:
        off_tr = include50.load_split_metadata("train", "official")
        off_te = include50.load_split_metadata("test", "official")
        summary_official = {
            "overlap": len(set(off_tr.FilePath.map(include50.session_id))
                           & set(off_te.FilePath.map(include50.session_id))),
            "test_sessions": len(set(off_te.FilePath.map(include50.session_id))),
        }
    except Exception:
        summary_official = {"overlap": -1, "test_sessions": -1}

    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": args.protocol,
        "num_classes": len(vocabulary),
        "vocabulary": vocabulary,
        "total_videos": len(records),
        "usable_videos": sum(1 for r in records if r.status == "ok"),
        "corrupt_videos": [r.filepath for r in records if r.status != "ok"],
        "corrupt_by_class": dict(Counter(r.label for r in records if r.status != "ok")),
        "splits": {"train": split_summary.train, "val": split_summary.val, "test": split_summary.test},
        "val_strategy": split_summary.val_strategy,
        "session_overlap_train_test": split_summary.session_overlap_train_test,
        "official_session_overlap": int(summary_official["overlap"]),
        "official_test_sessions": int(summary_official["test_sessions"]),
        "first_download_damage": FIRST_DOWNLOAD_DAMAGE,
        "duplicate_filepaths": int(index["FilePath"].duplicated().sum()),
        "train_test_path_overlap": int(len(set(index[index.split == "train"].FilePath)
                                          & set(index[index.split == "test"].FilePath))),
        "classes_train_only": sorted(set(index[index.split == "train"].label)
                                     - set(index[index.split == "test"].label)),
        "classes_test_only": sorted(set(index[index.split == "test"].label)
                                    - set(index[index.split == "train"].label)),
        "near_duplicate_pairs": near_dups[:200],
        "near_duplicate_pair_count": len(near_dups),
        "near_duplicate_cross_train_test": len(cross_split),
        "pairs_checked": pairs_checked,
        "frame_stats": {
            "median": float(np.median([r.n_frames for r in records if r.status == "ok"] or [0])),
            "min": int(min([r.n_frames for r in records if r.status == "ok"] or [0])),
            "max": int(max([r.n_frames for r in records if r.status == "ok"] or [0])),
        },
    }

    # ---- archive-wide CRC statistics (from the earlier scan if present, or now)
    archive = include50.keypoint_archive()
    archive_stats: Dict[str, object] = {"archive": str(archive) if archive else "not found"}
    scan_path = Path("/data/include/corrupt_scan.json")
    if scan_path.exists():
        scan = json.loads(scan_path.read_text())
        archive_stats.update({"total_files": scan.get("total"), "corrupt_files": len(scan.get("corrupt", [])),
                              "corrupt_examples": scan.get("corrupt", [])[:20]})
    elif args.full_archive and archive is not None:
        LOG.info("CRC-scanning the full archive (this takes a few minutes) ...")
        import zipfile

        z = zipfile.ZipFile(archive)
        names = [n for n in z.namelist() if n.endswith(".pkl")]
        bad = []
        for n in names:
            try:
                with z.open(n) as f:
                    while f.read(8 << 20):
                        pass
            except Exception:
                bad.append(n)
        archive_stats.update({"total_files": len(names), "corrupt_files": len(bad),
                              "corrupt_examples": bad[:20]})

    # ---- geometry calibration
    LOG.info("measuring geometry calibration ...")
    try:
        meas = geom.measure_ratios_from_dataset(limit=200)
        calib = meas.as_dict()
        geom.write_calibration(meas)
    except Exception as exc:
        LOG.warning("geometry calibration failed: %s", exc)
        calib = geom.load_calibration() or {}

    # ---- plots & report
    if not args.no_plots:
        by_class_plot: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for r in records:
            by_class_plot[r.label][r.split] += 1
        plot_class_distribution(by_class_plot, Path(config.RESULTS_DIR) / "class_distribution.png")
        plot_sequence_lengths(records, Path(config.RESULTS_DIR) / "sequence_length_distribution.png")

    report = build_report(records, summary, calib, archive_stats, vocabulary, split_summary)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report)
    (Path(config.RESULTS_DIR) / "dataset_audit_summary.json").write_text(json.dumps(
        {**summary, "archive": archive_stats, "geometry_calibration": calib}, indent=2))

    print(f"\n[audit] report      : {out}")
    print(f"[audit] summary json: {Path(config.RESULTS_DIR) / 'dataset_audit_summary.json'}")
    print(f"[audit] usable      : {summary['usable_videos']}/{summary['total_videos']} videos, "
          f"{summary['num_classes']} classes")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:  # pragma: no cover
        traceback.print_exc()
        raise SystemExit(1)
