---
title: Real Time ISL Command Recognition
emoji: 🤟
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Real-Time Indian Sign Language Command Recognition with Temporal Deep Learning

A final-year AI capstone: recognising a **fixed vocabulary of isolated ISL commands**
from a webcam or a video file, with three temporal deep-learning models trained and
compared under identical conditions.

> **Scope, read this first.** This is an **isolated sign/command recogniser** on the
> AI4Bharat **INCLUDE-50** subset. It is **not** a natural-language ISL translator, it
> does not model grammar or sentence structure, and it does **not** recognise "all of
> Indian Sign Language". It is a research prototype, not a certified accessibility
> tool. See [`docs/limitations.md`](docs/limitations.md).

---

## What it does

```
webcam / video file / dataset clip
        │
        ▼
MediaPipe landmarks: 33 pose + 21 left hand + 21 right hand = 75 landmarks
                     → (x, y, z) = 225 features per frame, shoulder-normalised
        │
        ▼
temporal buffer → fixed-length sequence (default 30 frames, configurable)
        │
        ▼
three models trained on identical data / split / seed / augmentation:
        LSTM(128→64)                                       baseline A
        GRU(128→64)                                        baseline B
        GRU(128) → GRU(64) → Multi-Head Self-Attention → temporal pooling → Dense → Softmax   proposed C
        │
        ▼
softmax → confidence gate (UNCERTAIN below threshold) → 3-frame majority vote
        → stable-prediction check → duplicate suppression + cooldown
        │
        ▼
caption text → optional speech (ON/OFF) → human ACCEPT / CORRECT / REJECT
        │
        ▼
JSONL audit log → error-review view → optional bounded personalisation
```

## Verified results (produced by this repository, not quoted from a paper)

Primary **leak-free** split (`session_disjoint`: 652 train / 152 val / **154 test**,
whole recording clusters kept on one side of the split):

| model | params | accuracy | macro F1 | weighted F1 | latency (ms) |
|---|---|---|---|---|---|
| LSTM(128→64) | 245,426 | 0.6623 | 0.5969 | 0.6025 | 2.76 |
| GRU(128→64) | 188,338 | 0.6299 | 0.5588 | 0.5643 | 2.43 |
| **GRU + Multi-Head Attention (proposed)** | 263,026 | **0.7727** | **0.7182** | **0.7255** | 2.73 |

Latency here is model-only for one sequence after warm-up; it excludes MediaPipe,
capture, networking and rendering. Measured batch-1 model throughput is 362 / 412 /
366 sequences/s respectively. Live end-to-end latency is separately shown in the UI.

AI4Bharat's official split reuses recording sessions across train/test, but an
official-split result is not present in the current evaluation bundle. Do not quote a
protocol-to-protocol accuracy gap until that evaluation is regenerated.

Robustness (14 landmark/sequence perturbations, real models): dropping 40% of frames
costs GRU+MHA about 0.026 accuracy; removing both hands reduces it to 0.117. These
are landmark-space tests, not camera-lighting tests. At the shipped confidence gate
0.70 the validation sweep covers 58% of clips at 85% selective accuracy. Lighting
robustness, signer-independent accuracy and participant accessibility feedback remain
unevaluated and are documented as gaps.

Full tables, failure lists and methods: [`docs/evaluation.md`](docs/evaluation.md),
[`docs/model_card.md`](docs/model_card.md).

---

## Verified in this checkout

Everything below was executed here and is reproducible with the commands shown
(the machine: 2 vCPU, CPU-only, Python 3.13.14, TensorFlow 2.21).

| check | command | observed result |
|---|---|---|
| test suite | `python -m pytest tests -q` | **165 passed, 0 skipped** in the 2026-10-05 run, before the source-folder cleanup in this revision; rerun after checkout |
| data / model / API / logging tests | `python -m pytest tests/test_api.py tests/test_personalization.py -q` | API: validation, upload safety, health, predict, feedback, personalisation all pass |
| dataset audit | `python run.py audit` | 958/958 clips usable, 0 corrupt, 50 classes, protocol-leakage table |
| preprocessing | `python training/preprocess.py` | `sequences_session_disjoint.npz` 1304/152/154, 0 clips skipped |
| training | `python training/train_{lstm,gru,gru_mha}.py` | 245,426 / 188,338 / 263,026 params; best val acc 0.6513 / 0.6513 / **0.7697** |
| evaluation | `python run.py evaluate` (+ `--protocol official`) | tables in §results above and `results/` |
| robustness | `python run.py robustness` | 14 conditions × 3 models, both protocols |
| threshold study | `python run.py threshold` | `results/threshold_sweep.csv` |
| live API | `uvicorn backend.main:app` + `curl` | `/health` ok (gru_mha, 50 classes); `/predict` on a test clip → `BIRD` 0.759 CONFIDENT with `event_id` logged |
| browser dashboard | `python run.py api` | served by FastAPI at `/`; API docs at `/docs` |
| batch inference | `python run.py batch --from-dataset --split test --limit 12 --model gru_mha` | 12/12 clips processed, raw accuracy 0.75 on this small sample, uncertain rate 66.7%, mean latency 23.3 ms (P95 112.0 ms) |
| live camera smoke check | `python run.py realtime --camera 0 --no-speech --no-window --seconds 8` | camera opened at about 28.5 FPS; pose detected, no hands/sign presented, so no prediction was emitted; this does not validate live sign accuracy |
| environment check | `python run.py doctor` | all dependencies + assets OK |
| camera diagnosis | `python run.py camera` | Windows laptop camera 0 opened at 640×480, 30 FPS; use this check to inspect camera availability on another machine |

The full test run above predates the folder cleanup in this revision. Re-run the suite
after checkout. The live-camera smoke check did not include a performed sign, and the
Docker image has not been built on this machine; live sign accuracy and signer-independent
performance remain unverified.

## Documentation map

| document | contents |
|---|---|
| [`docs/dataset_audit.md`](docs/dataset_audit.md) | measured audit of INCLUDE-50 on this machine, incl. the split-leakage comparison |
| [`docs/evaluation.md`](docs/evaluation.md) | protocol, results per model, per-class analysis, threshold study, robustness, what is *not* measured |
| [`docs/model_card.md`](docs/model_card.md) | architectures, training data, intended use, ethics, reproduction |
| [`docs/system_card.md`](docs/system_card.md) | deployed system, data handling/privacy, oversight, failure modes |
| [`docs/architecture.md`](docs/architecture.md) | repository layout, data flow, contracts, extension points |
| [`docs/setup.md`](docs/setup.md) | install, dataset, all commands, Docker + webcam notes, troubleshooting |
| [`docs/limitations.md`](docs/limitations.md) | scope limits, measured dataset limits, claims deliberately not made |
| [`docs/evaluation_artifact_map.md`](docs/evaluation_artifact_map.md) | evaluation protocol and artefact map |

---

## Where to run this

Everything is plain Python — there is no build step. Run every command **from the
project root** (the folder containing `run.py`), because all paths in `config.py`
(`models/`, `datasets/processed/`, `results/`, `logs/`) are relative to it.

```
isl-recognition/          <- project root: run commands here
├── run.py                <- single entry point for every stage
├── config.py             <- all settings (relative paths, env-overridable via ISL_*)
├── datasets/
│   ├── raw/              <- put INCLUDE-50 videos here (or the archives: see docs/setup.md)
│   ├── processed/        <- generated tensors (rebuild with `python run.py preprocess`)
│   └── metadata/         <- authoritative split CSVs (already included)
├── models/               <- trained weights + class mapping + MediaPipe .task bundles
├── results/              <- every CSV/PNG the report cites
├── frontend/             <- browser dashboard
├── backend/              <- API, inference, preprocessing and schemas
├── training/             <- dataset audit, model training and evaluation
├── tests/ docs/ datasets/ models/ results/ scripts/ .github/
```

```bash
cd isl-recognition                 # 1. enter the project root
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run.py doctor               # 2. verify: deps, MediaPipe bundles, dataset, webcam
python run.py api                  # 3. open the live dashboard at http://localhost:8000
```

* **Your own machine** — clone or download the GitHub repository, then follow the
  three steps above. Trained weights and MediaPipe task bundles are included for
  inference; raw recordings and local audit logs are intentionally excluded.
  `python run.py doctor` reports any missing local assets.
* **Docker** — no local Python needed: `docker compose up --build` (API and dashboard :8000,
  no webcam; see the limitation note below).

## Quick start

```bash
pip install -r requirements.txt
python run.py doctor                  # environment + asset check
python run.py info                    # configuration + what exists on disk

python run.py audit                   # dataset audit      -> docs/dataset_audit.md
python run.py preprocess              # tensors            -> datasets/processed/*.npz
python run.py train                   # all three models   -> models/{lstm,gru,gru_mha}/
python run.py evaluate                # comparison         -> results/*.csv, *.png
python run.py robustness              # perturbation study -> results/robustness_results.csv
python run.py threshold               # gate sweep         -> results/threshold_sweep.csv

python run.py api                     # FastAPI            -> http://localhost:8000/docs
# browser dashboard (live camera + API): http://localhost:8000
python run.py realtime                # webcam recognition (local machine)
python run.py batch --from-dataset --split test --limit 20
python run.py test                    # pytest suite
```

`run.py` forwards flags to the scripts, e.g. `python run.py train --model gru_mha
--epochs 40`, `python run.py evaluate --protocol official`.

### Docker

```bash
docker compose up --build             # API + dashboard :8000
```

**Webcam limitation:** live capture needs a **capture device** *and* (for the preview
window) a **display**. Containers, servers, WSL/SSH sessions and this project's cloud
sandbox have neither, and a sandboxed preview iframe cannot grant browser camera
access, so live video cannot work there by construction. Diagnose any machine with:

```bash
python run.py camera     # devices, indices 0-3, DISPLAY, GUI support, matching fix
```

Fallbacks that always work: `python run.py batch --from-dataset`,
`python run.py batch --video clip.mp4`, or the UI's upload / dataset-demo tab.
On a desktop machine, `python backend/inference/realtime.py` gives full camera rate (add
`--no-window` for terminal-only operation, `--camera 1` for a second camera).
Full causes/fixes table: [`docs/setup.md`](docs/setup.md) §7.

---

## API (FastAPI)

| method | path | purpose |
|---|---|---|
| GET | `/health` | liveness + whether a model could be loaded (never fakes `ok`) |
| GET | `/model-info` | models on disk, params, class list, preprocessing + gate settings, TTS status |
| GET | `/metrics` | live audit-log statistics (latency, uncertain rate, human actions) |
| GET | `/config`, `/classes`, `/dataset/clips` | runtime settings, the 50 class names, demo clips |
| POST | `/predict` | prediction from an **uploaded video** (multipart) or JSON (`features` / `dataset_clip`) |
| POST | `/feedback` | ACCEPT / CORRECT / REJECT for a logged event |
| POST | `/personalize`, GET `/profiles` | bounded profiles: vocabulary subset, ±0.2 threshold band, calibration, optional fine-tune |
| GET | `/audit/events`, `/audit/pairs` | error review with filters (low confidence, rejected, corrected, class, model, status) |

Response shape of `/predict`: `{sign, confidence, status, latency_ms, model, raw_sign,
raw_confidence, stable, caption, threshold, event_id, top_k}`. If no trained model
exists the endpoint returns **503 with the exact training command** — it never returns a
fabricated label.

## Frontend

The main dashboard is the browser app in `frontend/`, served by FastAPI at
`http://localhost:8000`. It uses the same API for live capture, video upload, dataset
demos, history, human review and model metrics.

## Audit logging

`logs/predictions.jsonl` — one record per emission with `event_id, timestamp, model,
predicted_class, confidence, status, latency_ms, fps, human_action, corrected_class,
source, profile`. `logs/feedback/feedback.jsonl` holds human verdicts joined by
`event_id`. Personalisation stores **features**, never video; uploads are processed in a
temp directory and deleted unless `ISL_PERSIST_UPLOADS=true`.

---

## Project layout

```
config.py · run.py · requirements.txt · .env.example · Dockerfile · docker-compose.yml
frontend/        browser dashboard (HTML, CSS and JavaScript)
backend/         FastAPI API, inference, preprocessing, schemas and personalisation
training/        dataset audit, model training, evaluation and robustness
datasets/        raw recordings (local), processed sequences and versioned metadata
models/          LSTM, GRU, GRU+MHA, class maps and MediaPipe task bundles
results/         evaluation tables/plots, robustness and threshold-sweep evidence
docs/            architecture, setup, audit, evaluation and model/system cards
tests/           automated API, data, model and inference tests
scripts/         camera diagnostics and MediaPipe asset setup
.github/         continuous-integration workflow
```

## Fair-comparison contract

All three models share: the same processed tensors, the same split and class mapping,
the same seed (42), the same offline augmentation, the same callbacks, epochs, batch
size, loss and label smoothing, and the same frozen test set with no test-time
augmentation. Each model directory stores `class_mapping.json`,
`preprocessing_config.json` and `training_history.json` so a run can be audited.

## Tests

```bash
python run.py test          # or: python -m pytest tests -v
```

Tests needing the dataset or trained models skip with an explicit reason when the
artefact is missing, so the suite is honest on a fresh checkout. Metrics code is checked
against hand-computed values; the API tests cover validation, upload safety, health,
predict, feedback and personalisation.

## Dataset and citation

Training data: **AI4Bharat INCLUDE** (Sridhar, Ganesan, Kumar, Khapra, *INCLUDE: A
Large Scale Dataset for Indian Sign Language Recognition*, ACM Multimedia 2020,
DOI [10.1145/3394171.3413528](https://doi.org/10.1145/3394171.3413528)), INCLUDE-50
subset, official pre-extracted MediaPipe keypoints. Landmark extraction uses MediaPipe
Tasks (pose + hands). INCLUDE publishes **no signer ids**, which is why the primary
evaluation is recording-cluster-disjoint rather than signer-disjoint
([`docs/limitations.md`](docs/limitations.md)).

## Responsible-use summary

Advisory output only; a human can override every prediction; confidence-gated so the
system says `UNCERTAIN` instead of guessing; performance is unmeasured for other
signers, dialects, lighting or cameras; no deaf-community validation was performed; the
project explicitly does not claim to be a sign-language translator. Privacy-preserving
by default: no video retention, local JSONL logs, secrets in `.env` only.
