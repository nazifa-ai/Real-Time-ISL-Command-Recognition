"""
Shared pytest fixtures.

The suite runs against the *real* artefacts of this checkout wherever they exist
(the frozen INCLUDE-50 index, the processed tensors, the trained models) and skips
the corresponding tests with an explicit reason when they do not.  Nothing is
mocked into pretending a model or dataset is present.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config  # noqa: E402


# --------------------------------------------------------------------------
# paths
# --------------------------------------------------------------------------
@pytest.fixture(scope="session")
def root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def vocabulary_path() -> Path:
    return Path(config.METADATA_DIR) / "vocabulary.json"


@pytest.fixture(scope="session")
def processed_path() -> Path:
    return Path(config.PROCESSED_DIR) / f"sequences_{config.SPLIT_PROTOCOL}.npz"


@pytest.fixture(scope="session")
def has_processed(processed_path: Path) -> bool:
    return processed_path.exists()


# --------------------------------------------------------------------------
# data fixtures
# --------------------------------------------------------------------------
@pytest.fixture(scope="session")
def vocabulary() -> list:
    p = Path(config.METADATA_DIR) / "vocabulary.json"
    if not p.exists():
        pytest.skip(f"vocabulary missing ({p}) - run training/fetch_metadata.py")
    return [c["label"] for c in json.loads(p.read_text())["classes"]]


@pytest.fixture(scope="session")
def index_df():
    p = Path(config.METADATA_DIR) / f"index_{config.SPLIT_PROTOCOL}.csv"
    if not p.exists():
        pytest.skip(f"split index missing ({p}) - run backend.preprocessing.include50.build_index()")
    import pandas as pd

    return pd.read_csv(p)


@pytest.fixture(scope="session")
def processed(processed_path: Path):
    if not processed_path.exists():
        pytest.skip(f"processed tensors missing ({processed_path}) - run training/preprocess.py")
    data = np.load(processed_path, allow_pickle=True)
    return {k: data[k] for k in data.files}


@pytest.fixture(scope="session")
def test_tensor(processed):
    return processed["X_test"], processed["y_test"]


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    return np.random.default_rng(config.RANDOM_SEED)


@pytest.fixture(scope="session")
def sample_sequence() -> np.ndarray:
    """Deterministic (30, 75, 3) landmark clip with plausible body geometry."""
    rng = np.random.default_rng(0)
    coords = rng.normal(0, 0.02, size=(config.SEQUENCE_LENGTH, 75, 3)).astype(np.float32)
    # place shoulders 0.4 apart around the origin, hips below
    coords[:, 11] = np.array([-0.20, 0.10, 0.0])
    coords[:, 12] = np.array([0.20, 0.10, 0.0])
    coords[:, 23] = np.array([-0.14, -0.22, 0.0])
    coords[:, 24] = np.array([0.14, -0.22, 0.0])
    return coords


@pytest.fixture()
def trained_model():
    """The default trained model, or a skip explaining how to produce it."""
    import tensorflow as tf

    name = config.DEFAULT_MODEL
    path = Path(config.MODEL_PATHS[name]) / "model.keras"
    if not path.exists():
        pytest.skip(f"trained model missing ({path}) - run: python training/train_{name}.py")
    return name, tf.keras.models.load_model(str(path))


@pytest.fixture()
def all_trained_models():
    import tensorflow as tf

    out = {}
    for name in config.AVAILABLE_MODELS:
        path = Path(config.MODEL_PATHS[name]) / "model.keras"
        if path.exists():
            out[name] = tf.keras.models.load_model(str(path))
    if not out:
        pytest.skip("no trained models on disk - run training/train_*.py")
    return out


# --------------------------------------------------------------------------
# isolation fixtures
# --------------------------------------------------------------------------
@pytest.fixture()
def temp_logs(tmp_path, monkeypatch):
    """Redirect audit logging into a temporary directory (never touch real logs)."""
    logs = tmp_path / "logs"
    (logs / "feedback").mkdir(parents=True, exist_ok=True)
    monkeypatch.setitem(config.LOG_PATHS, "predictions", logs / "predictions.jsonl")
    monkeypatch.setitem(config.LOG_PATHS, "feedback", logs / "feedback" / "feedback.jsonl")
    monkeypatch.setitem(config.LOG_PATHS, "audit", logs / "audit.jsonl")
    return logs


@pytest.fixture()
def api_client(temp_logs):
    """TestClient bound to the FastAPI app.

    Uses the real engines on disk but writes audit records into a temporary
    directory, so running the suite never pollutes the application's logs.
    """
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from backend import main as api

    api.clear_engine_cache()
    with TestClient(api.app) as client:
        yield client
    api.clear_engine_cache()
