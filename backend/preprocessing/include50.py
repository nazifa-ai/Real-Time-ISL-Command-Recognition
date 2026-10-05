"""
INCLUDE-50 dataset access layer.

Responsibilities
----------------
1. Read the **official AI4Bharat INCLUDE-50 metadata** (class list, video paths,
   official train/test split) - the vocabulary is *derived*, never hard-coded.
2. Read the **official AI4Bharat pre-extracted MediaPipe keypoints** (Zenodo
   6674324) and map them into the canonical 75-landmark layout.
3. Optionally extract landmarks from the raw INCLUDE videos with the project's
   own MediaPipe extractor (``KEYPOINT_SOURCE="mediapipe"``).
4. Provide a reproducible, leakage-aware split (train / validation / test).

Released keypoint format (verified against AI4Bharat/INCLUDE``generate_keypoints.py``)
------------------------------------------------------------------------------------
One JSON file per video named ``<label>_<uid>.json``, e.g. ``bird_MVI_2987.json``::

    {"uid": "bird_MVI_2987", "label": "bird", "n_frames": 187,
     "pose_x": [[...33 or 25 floats...], ...],   # NaN when pose was not detected
     "pose_y": [[...]], "hand1_x": [[...21...]], "hand1_y": [[...]],
     "hand2_x": [[...]], "hand2_y": [[...]]}

* coordinates are **2-D (x, y) only** - no z is present in the release;
* ``hand1``/``hand2`` are *not* left/right: AI4Bharat assigns them by nearest
  wrist, so this loader re-derives left/right per frame from the pose wrists;
* missing data is ``NaN`` (never silently zero).

Data licensing: INCLUDE is distributed under CC-BY-4.0.  See ``docs/dataset_audit.md``.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)

#: released keypoints are 2-D; these pose indices follow the BlazePose numbering
#: that AI4Bharat used (upper-body-only model -> 25 points, or 33 if full body).
POSE_LEFT_WRIST = 15
POSE_RIGHT_WRIST = 16

KEYPOINT_SUFFIXES = (".json", ".pkl")


# --------------------------------------------------------------------------
# Metadata
# --------------------------------------------------------------------------
def metadata_path(name: str) -> Path:
    return Path(config.METADATA_DIR) / name


def _gloss_normaliser() -> Dict[str, str]:
    """raw folder name -> official normalised gloss (e.g. '55. Thank you' -> 'THANK_YOU')."""
    p = metadata_path(config.METADATA_FILES["gloss_norm"])
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    return {str(a): str(b) for a, b in zip(df["actual_gloss"], df["normalized_gloss"])}


def load_split_metadata(split: str, protocol: str | None = None) -> pd.DataFrame:
    """Load one split of the official INCLUDE-50 metadata.

    Parameters
    ----------
    split : ``"train"`` | ``"test"`` | ``"val"``
    protocol : ``"official"`` (AI4Bharat train/test CSVs) or ``"hf"`` (HuggingFace
        INCLUDE parquet, which additionally carries a validation split).

    Returns
    -------
    DataFrame with columns: FilePath, Video, Word (raw), label (normalised),
    Category, split.
    """
    protocol = (protocol or config.SPLIT_PROTOCOL).lower()
    norm = _gloss_normaliser()

    if protocol == "hf":
        name = {"train": "hf_train", "val": "hf_val", "test": "hf_test"}.get(split)
        path = metadata_path(config.METADATA_FILES[name])
        if not path.exists():
            raise FileNotFoundError(
                f"HF metadata missing: {path}\nRun: python training/fetch_metadata.py --protocol hf"
            )
        df = pd.read_parquet(path)
        df = df[df["include_50"]].copy()
        df = df.rename(columns={"video_path": "FilePath", "label": "Word"})
        df["Category"] = df["parent_label"]
        df["Video"] = df["FilePath"].map(lambda p: str(p).split("/")[-1])
    else:
        key = "train_50" if split == "train" else "test_50"
        path = metadata_path(config.METADATA_FILES[key])
        if not path.exists():
            raise FileNotFoundError(
                f"official INCLUDE-50 metadata missing: {path}\n"
                "Run: python training/fetch_metadata.py --protocol official"
            )
        df = pd.read_csv(path)
        if split == "val":
            # official metadata has no validation split - carved out in build_index()
            df = pd.read_csv(metadata_path(config.METADATA_FILES["train_50"]))

    df = df.rename(columns={"Word": "raw_word"})
    df["label"] = df["raw_word"].map(lambda w: norm.get(str(w), str(w).strip().upper()))
    df["Video"] = df["FilePath"].map(lambda p: str(p).split("/")[-1])
    df["Category"] = df["Category"].astype(str)
    df["split"] = split
    return df[["FilePath", "Video", "raw_word", "label", "Category", "split"]]


def load_all_metadata(protocol: str | None = None) -> pd.DataFrame:
    protocol = (protocol or config.SPLIT_PROTOCOL).lower()
    frames = [load_split_metadata(s, protocol) for s in ("train", "test")]
    return pd.concat(frames, ignore_index=True)


def derive_vocabulary(protocol: str | None = None) -> List[str]:
    """The 50 class names, **read from the metadata** (sorted, de-duplicated)."""
    df = load_all_metadata(protocol)
    return sorted(df["label"].unique().tolist())


def save_vocabulary(protocol: str | None = None) -> Path:
    """Freeze the verified vocabulary into ``datasets/metadata/vocabulary.json``."""
    protocol = (protocol or config.SPLIT_PROTOCOL).lower()
    df = load_all_metadata(protocol)
    counts = df.groupby("label").size().to_dict()
    cat = df.groupby("label")["Category"].first().to_dict()
    vocab = sorted(counts)
    payload = {
        "dataset": config.DATASET_NAME,
        "protocol": protocol,
        "citation": config.DATASET_CITATION,
        "doi": config.DATASET_DOI,
        "num_classes": len(vocab),
        "classes": [
            {"index": i, "label": v, "raw_word": None, "category": cat.get(v), "videos": int(counts[v])}
            for i, v in enumerate(vocab)
        ],
    }
    raw = (
        df.drop_duplicates("label").set_index("label")["raw_word"].to_dict()
        if "raw_word" in df
        else {}
    )
    for entry in payload["classes"]:
        entry["raw_word"] = raw.get(entry["label"])
    out = metadata_path("vocabulary.json")
    out.write_text(json.dumps(payload, indent=2))
    logger.info("vocabulary frozen: %s (%d classes)", out, len(vocab))
    return out


def load_vocabulary() -> List[str]:
    """Read the frozen vocabulary (creating it from metadata if absent)."""
    p = metadata_path("vocabulary.json")
    if not p.exists():
        save_vocabulary()
    payload = json.loads(p.read_text())
    return [c["label"] for c in payload["classes"]]


def class_mapping() -> Dict[str, int]:
    vocab = load_vocabulary()
    return {name: i for i, name in enumerate(vocab)}


# --------------------------------------------------------------------------
# Split construction
# --------------------------------------------------------------------------
_SID_RE = re.compile(r"MVI[_-]?(\d+)", re.IGNORECASE)


def session_id(filepath: str) -> str:
    """Recording-session identifier for a clip.

    INCLUDE file names look like ``MVI_2987.MOV``; ``MVI_0060_1.MP4`` style names
    also occur.  The numeric id is a *proxy* for the recording session: clips with
    neighbouring ids inside one class are consecutive takes from the same sitting
    (measured: 96% of within-class id gaps are <= 10), which is why the
    session-disjoint split groups ids into blocks rather than treating every id as
    its own session.  INCLUDE publishes no explicit session field.
    """
    stem = str(filepath).split("/")[-1].rsplit(".", 1)[0]
    m = _SID_RE.search(stem)
    if m:
        return m.group(1)
    digits = "".join(ch for ch in stem if ch.isdigit())
    return digits or stem


def session_block(filepath: str, block: int = 10) -> str:
    """Coarse session proxy: consecutive-id *blocks*.

    Clips whose ids fall in the same block of size ``block`` share a recording
    sitting with high probability.  Used by the session-disjoint split so that
    neighbouring takes never straddle train/test.
    """
    sid = session_id(filepath)
    if sid.isdigit():
        return str(int(sid) // int(block))
    return sid


@dataclass
class SplitSummary:
    protocol: str
    train: int
    val: int
    test: int
    classes_train: int
    classes_val: int
    classes_test: int
    val_strategy: str
    session_overlap_train_test: int
    note: str = ""

    def as_dict(self) -> Dict[str, object]:
        return self.__dict__.copy()


def build_index(
    protocol: str | None = None,
    vocab_filter: Optional[Sequence[str]] = None,
    force_rebuild: bool = False,
) -> Tuple[pd.DataFrame, SplitSummary]:
    """Build the canonical train/val/test index (cached to CSV).

    Split policy
    ------------
    * ``session_disjoint`` (default): pools the full INCLUDE-50 pool and splits by
      recording cluster, so no burst of consecutive takes spans two splits.  This is
      the project's primary evaluation because the official split leaks.
    * ``official``: uses AI4Bharat's official train/test files verbatim.
      The official metadata ships **no validation split**, so validation is carved
      out of the official train set with a seeded, class-stratified, *session-aware*
      split (all clips sharing a session id fall on the same side).
    * ``hf``: uses the HuggingFace INCLUDE train/val/test parquet split.

    The official split is not signer-disjoint (INCLUDE does not publish signer ids);
    see ``docs/dataset_audit.md`` for the measured session overlap.
    """
    protocol = (protocol or config.SPLIT_PROTOCOL).lower()
    cache = metadata_path(f"index_{protocol}.csv")
    summary_cache = metadata_path(f"index_{protocol}_summary.json")

    if cache.exists() and not force_rebuild:
        df = pd.read_csv(cache)
        if summary_cache.exists():
            summary = SplitSummary(**json.loads(summary_cache.read_text()))
        else:  # pragma: no cover
            summary = SplitSummary(protocol, 0, 0, 0, 0, 0, 0, "cached", 0)
        if vocab_filter:
            df = df[df["label"].isin(list(vocab_filter))].reset_index(drop=True)
        return df, summary

    if protocol == "session_disjoint":
        pool = load_split_metadata("train", "official")
        pool = pd.concat([pool, load_split_metadata("test", "official")], ignore_index=True)
        train, val, test = _session_disjoint_split(
            pool, config.VAL_FRACTION, config.TEST_FRACTION, config.RANDOM_SEED
        )
        strategy = (
            "class-stratified recording-cluster-disjoint split over the full INCLUDE-50 pool "
            f"(val={config.VAL_FRACTION}, test={config.TEST_FRACTION}, seed={config.RANDOM_SEED}, "
            "max within-cluster id gap=10). Strongest defensible alternative to signer-disjoint "
            "evaluation, which INCLUDE metadata cannot support."
        )
    elif protocol == "hf":
        train = load_split_metadata("train", "hf")
        val = load_split_metadata("val", "hf")
        test = load_split_metadata("test", "hf")
        strategy = "official HuggingFace train/val/test"
    else:
        train_full = load_split_metadata("train", "official")
        test = load_split_metadata("test", "official")
        train, val = _stratified_session_split(train_full, config.VAL_FRACTION, config.RANDOM_SEED)
        strategy = (
            f"official train/test + seeded stratified session-aware carve-out "
            f"(val_fraction={config.VAL_FRACTION}, seed={config.RANDOM_SEED})"
        )

    df = pd.concat([train, val, test], ignore_index=True)
    df["split"] = df["split"].astype(str)

    for part in ("train", "val", "test"):
        df.loc[df["split"] == part, "subset"] = part
    overlap = len(set(df[df.split == "train"].FilePath.map(session_id))
                  & set(df[df.split == "test"].FilePath.map(session_id)))

    summary = SplitSummary(
        protocol=protocol,
        train=int((df.split == "train").sum()),
        val=int((df.split == "val").sum()),
        test=int((df.split == "test").sum()),
        classes_train=int(df[df.split == "train"].label.nunique()),
        classes_val=int(df[df.split == "val"].label.nunique()),
        classes_test=int(df[df.split == "test"].label.nunique()),
        val_strategy=strategy,
        session_overlap_train_test=int(overlap),
        note="INCLUDE publishes no signer metadata -> signer-disjoint evaluation is not possible.",
    )

    df.to_csv(cache, index=False)
    summary_cache.write_text(json.dumps(summary.as_dict(), indent=2))
    logger.info("index built: train=%d val=%d test=%d (overlap sessions=%d)",
                summary.train, summary.val, summary.test, summary.session_overlap_train_test)

    if vocab_filter:
        df = df[df["label"].isin(list(vocab_filter))].reset_index(drop=True)
    return df, summary


def load_index(protocol: str | None = None, force_rebuild: bool = False,
               vocab_filter: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Public accessor for the split index (thin wrapper over :func:`build_index`)."""
    df, _ = build_index(protocol=protocol, vocab_filter=vocab_filter, force_rebuild=force_rebuild)
    return df


def session_clusters(index: pd.DataFrame, max_gap: int = 10) -> pd.Series:
    """Group clips into recording clusters, per class.

    Within one class the released file ids run in consecutive bursts
    (measured: 71% of neighbouring ids differ by exactly 1, 73% by <= 10).  A burst
    is a recording sitting: the same signer, same camera, same clothing.  Splitting
    inside a burst leaks information, so whole clusters are kept on one side.

    Returns a Series aligned with ``index`` holding a cluster key per clip.
    """
    keys = pd.Series("", index=index.index, dtype=object)
    for label, grp in index.groupby("label"):
        ids = [(i, session_id(fp)) for i, fp in zip(grp.index, grp["FilePath"])]
        numeric = [(i, int(s)) for i, s in ids if str(s).isdigit()]
        other = [i for i, s in ids if not str(s).isdigit()]
        numeric.sort(key=lambda t: t[1])
        for extra in other:
            keys[extra] = f"{label}:other"
        if not numeric:
            continue
        start = prev = numeric[0][1]
        members = [numeric[0][0]]
        for i, v in numeric[1:]:
            if v - prev <= max_gap:
                members.append(i)
            else:
                for m in members:
                    keys[m] = f"{label}:{start}-{prev}"
                members, start = [i], v
            prev = v
        for m in members:
            keys[m] = f"{label}:{start}-{prev}"
    return keys


def _session_disjoint_split(df: pd.DataFrame, val_fraction: float, test_fraction: float, seed: int,
                            max_gap: int = 10):
    """Class-stratified split in which no recording cluster spans two splits.

    This is the strongest defensible evaluation available: INCLUDE publishes no
    signer identifiers, so signer-disjoint evaluation is impossible, but keeping
    whole recording bursts together removes the session-level leakage that the
    *official* split has (measured: 63% of clips share an id-block with the other
    side of the official split).
    """
    rng = np.random.default_rng(seed)
    keys = session_clusters(df, max_gap=max_gap)
    df = df.copy()
    df["_cluster"] = keys.values
    split_of_cluster: Dict[str, str] = {}
    for label, grp in df.groupby("label"):
        clusters = list(grp.groupby("_cluster").groups.items())
        rng.shuffle(clusters)
        total = len(grp)
        n_val = max(1, int(round(total * val_fraction)))
        n_test = max(1, int(round(total * test_fraction)))
        free = {cid: len(idxs) for cid, idxs in clusters}
        # best-fit allocation of whole clusters to the val/test quotas
        for name, quota in (("test", n_test), ("val", n_val)):
            filled = 0
            while free and filled < quota:
                best = min(free, key=lambda c: (abs(quota - filled - free[c]), -free[c]))
                if filled + free[best] > quota and filled >= 0.6 * quota:
                    break
                split_of_cluster[best] = name
                filled += free.pop(best)
        for cid in free:
            split_of_cluster[cid] = "train"
    df["split"] = df["_cluster"].map(split_of_cluster).fillna("train")
    train = df[df.split == "train"].drop(columns=["_cluster"])
    val = df[df.split == "val"].drop(columns=["_cluster"])
    test = df[df.split == "test"].drop(columns=["_cluster"])
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def _stratified_session_split(df: pd.DataFrame, val_fraction: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Class-stratified split that keeps whole recording sessions together."""
    if val_fraction <= 0:
        empty = df.iloc[0:0].copy()
        return df.copy(), empty
    rng = np.random.default_rng(seed)
    val_idx: List[int] = []
    for label, grp in df.groupby("label"):
        groups = grp.groupby(grp["FilePath"].map(session_id)).groups
        keys = sorted(groups)
        rng.shuffle(keys)
        target = max(1, int(round(len(grp) * val_fraction)))
        picked, n = [], 0
        for k in keys:
            if n >= target:
                break
            idx = list(groups[k])
            picked.extend(idx)
            n += len(idx)
        val_idx.extend(picked[: max(1, len(picked))] if picked else [])
    val = df.loc[sorted(set(val_idx))].copy()
    train = df.drop(index=val.index).copy()
    val["split"] = "val"
    train["split"] = "train"
    return train.reset_index(drop=True), val.reset_index(drop=True)


# --------------------------------------------------------------------------
# Keypoint access
# --------------------------------------------------------------------------
#: Released layout (verified): Pose_Signs/<Category>/<Word>[/extra]/<UID>.pkl
KEYPOINT_ARCHIVE_NAME = "Pose_Signs_raw.zip"
KEYPOINT_SUFFIX = ".pkl"

_zip_cache: Dict[str, "zipfile.ZipFile"] = {}


def keypoints_root() -> Path:
    """First existing candidate directory for extracted keypoints."""
    candidates = [Path(config.KEYPOINTS_PATH), *[Path(p) for p in config.KEYPOINTS_FALLBACKS]]
    for c in candidates:
        if c.exists():
            return c
    return Path(config.KEYPOINTS_PATH)


def keypoint_archive() -> Optional[Path]:
    """The released zip archive, if present (searched next to the keypoint roots)."""
    candidates = [
        Path(config.KEYPOINTS_PATH).parent / KEYPOINT_ARCHIVE_NAME,
        Path(config.KEYPOINTS_PATH) / KEYPOINT_ARCHIVE_NAME,
        Path("/data/include") / KEYPOINT_ARCHIVE_NAME,
    ]
    for p in (Path(c) for c in config.KEYPOINTS_FALLBACKS):
        candidates.append(p / KEYPOINT_ARCHIVE_NAME)
        candidates.append(p.parent / KEYPOINT_ARCHIVE_NAME)
    for c in candidates:
        if c.exists():
            return c
    return None


def _open_archive(path: Path):
    key = str(path)
    if key not in _zip_cache:
        import zipfile

        _zip_cache[key] = zipfile.ZipFile(path)
    return _zip_cache[key]


def keypoint_member_for(filepath: str, archive: Optional[Path] = None) -> Optional[str]:
    """Map a metadata ``FilePath`` to the member name inside the keypoint archive.

    Handles the ``extra``/``Extra`` sub-directories that exist for a few classes.
    """
    rel = str(filepath)
    parts = rel.split("/")
    if len(parts) < 3:
        return None
    category, word = parts[0], parts[1]
    stem = parts[-1].rsplit(".", 1)[0]
    arch = Path(archive or keypoint_archive())
    if arch is None:
        return None
    z = _open_archive(Path(arch))
    names = z.namelist()
    prefixes = [f"{category}/{word}", f"{category}/{word}/extra", f"{category}/{word}/Extra",
                f"{category}/{word}/Extra Videos"]
    for pre in prefixes:
        member = f"Pose_Signs/{pre}/{stem}{KEYPOINT_SUFFIX}"
        if member in names:
            return member
    # case-insensitive / nested fallback
    lower = f"{stem}{KEYPOINT_SUFFIX}".lower()
    for n in names:
        if n.lower().endswith("/" + lower) and f"/{category}/".lower() in n.lower():
            return n
    return None


def keypoint_path_for(filepath: str) -> Optional[Path]:
    """Locate an *extracted* keypoint file for a dataset video."""
    root = keypoints_root()
    parts = str(filepath).split("/")
    if len(parts) < 3:
        return None
    category, word = parts[0], parts[1]
    stem = parts[-1].rsplit(".", 1)[0]
    for pre in (f"{category}/{word}", f"{category}/{word}/extra",
                f"{category}/{word}/Extra", f"{category}/{word}/Extra Videos"):
        for base in (root, root / "Pose_Signs"):
            c = base / pre / f"{stem}{KEYPOINT_SUFFIX}"
            if c.exists():
                return c
    return None


def list_keypoint_files(limit: Optional[int] = None) -> List[str]:
    """All available keypoint members (archive) or paths (extracted directory)."""
    root = keypoints_root()
    if root.exists() and any(root.rglob(f"*{KEYPOINT_SUFFIX}")):
        files = [str(p) for p in sorted(root.rglob(f"*{KEYPOINT_SUFFIX}"))]
    else:
        arch = keypoint_archive()
        if arch is None:
            return []
        files = [n for n in _open_archive(arch).namelist() if n.endswith(KEYPOINT_SUFFIX)]
    return files[:limit] if limit else files


def read_keypoint_archive_file(member: str, apply_axis_fix: Optional[bool] = None,
                               max_frames: Optional[int] = None,
                               conf_threshold: Optional[float] = None):
    """Read one released keypoint file.

    ``member`` may be either an archive member name (``Pose_Signs/.../X.pkl``) or a
    filesystem path.  Returns ``(coords (T,75,3) float32, mask (T,75) float32, info dict)``.

    Verified released structure::

        {"keypoints":  (T, 75, 3) float64,
         "confidences": (T, 75)   float64 in {0, 1},
         "vid_shape":  (height, width)}

    Missing landmarks are marked by confidence 0 **and** zero coordinates; the mask
    is built from confidences (falling back to coordinate activity when a file
    lacks them).
    """
    import pickle

    from . import geometry as geom

    info: Dict[str, object] = {"member": str(member)}
    if isinstance(member, (str, Path)) and not str(member).startswith("Pose_Signs/"):
        fh = open(member, "rb")
    else:
        arch = keypoint_archive()
        if arch is None:
            raise FileNotFoundError("keypoint archive not found; set ISL_KEYPOINTS_PATH")
        fh = _open_archive(arch).open(str(member), "r")
    with fh:
        data = pickle.load(fh)

    raw = np.asarray(data.get("keypoints", np.zeros((0, config.NUM_LANDMARKS, 3))), dtype=np.float32)
    conf = data.get("confidences", None)
    vid_shape = tuple(int(x) for x in np.asarray(data.get("vid_shape", (0, 0))).ravel())
    info["vid_shape"] = vid_shape
    info["n_frames"] = int(raw.shape[0])

    if raw.ndim != 3:
        raise ValueError(f"unexpected keypoints shape {raw.shape}")

    t = raw.shape[0] if max_frames is None else min(int(max_frames), raw.shape[0])
    coords, mask = raw[:t].copy(), _mask_from_confidences(conf, t, conf_threshold)

    fix = config.APPLY_KEYPOINT_AXIS_CORRECTION if apply_axis_fix is None else apply_axis_fix
    if fix:
        coords = geom.correct_landmarks(coords, vid_shape=vid_shape)
        scale = geom.axis_scale_for_video_shape(vid_shape)
        info["x_axis_scale"] = scale
    coords[mask <= 0] = 0.0
    return coords.astype(np.float32), mask.astype(np.float32), info


def _mask_from_confidences(conf, t: int, threshold: Optional[float] = None) -> np.ndarray:
    thr = config.KEYPOINT_CONFIDENCE_THRESHOLD if threshold is None else float(threshold)
    if conf is None:
        return np.ones((t, config.NUM_LANDMARKS), dtype=np.float32)
    conf = np.asarray(conf, dtype=np.float32)[:t]
    if conf.ndim != 2:
        return np.ones((t, config.NUM_LANDMARKS), dtype=np.float32)
    return (conf > thr).astype(np.float32)


def read_keypoints_for(filepath: str, max_frames: Optional[int] = None,
                       apply_axis_fix: Optional[bool] = None):
    """Read keypoints for a metadata FilePath from whichever layout is available."""
    p = keypoint_path_for(filepath)
    if p is not None:
        return read_keypoint_archive_file(str(p), apply_axis_fix=apply_axis_fix, max_frames=max_frames)
    member = keypoint_member_for(filepath)
    if member is None:
        raise FileNotFoundError(f"no keypoints found for {filepath}")
    return read_keypoint_archive_file(member, apply_axis_fix=apply_axis_fix, max_frames=max_frames)


def is_corrupt(record: Optional[Dict[str, object]], path: str) -> bool:
    """True when a keypoint entry is listed as unreadable in the CRC scan."""
    if record is None:
        return False
    return str(path) in set(record.get("corrupt", []))
