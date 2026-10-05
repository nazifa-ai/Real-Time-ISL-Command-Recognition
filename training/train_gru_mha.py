#!/usr/bin/env python3
"""
PHASE 7 -- train the gru_mha model.

The architecture itself lives in training/models.py; this script only fixes which
one to train.  All models share training/train_common.py, so the seed, data,
augmentation, callbacks and hyper-parameters are identical by construction.

Usage
-----
    python training/train_gru_mha.py
    python training/train_gru_mha.py --epochs 60 --batch-size 32 --protocol session_disjoint
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.train_common import run_cli  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(run_cli("gru_mha"))
