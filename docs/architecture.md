# Architecture

How the repository is organised, what each module owns, and the contracts between
them. Everything is driven by `config.py`, so behaviour is changed by configuration,
not by editing code.

---

## 1. Repository layout

```
isl-recognition/
├── config.py                     single source of truth (env-overridable, ISL_* prefix)
├── run.py                        CLI entry point for every phase
├── requirements.txt              pinned, tested versions
├── .env.example  .gitignore      configuration template / secret + data exclusions
├── Dockerfile  docker-compose.yml  .dockerignore
├── .github/workflows/ci.yml      pytest + syntax/import checks + docker build
│
├── backend/preprocessing/
│   ├── include50.py              metadata readers, vocabulary freeze, splits, keypoint archive reader
│   ├── landmarks.py              canonical 75-landmark packing, presence masks, body-relative normalisation
│   ├── sequence.py               resample / pad / window / augment / perturbation operators
│   ├── geometry.py               axis calibration (x/y anisotropy fix), body-ratio statistics
│   └── mediapipe_extractor.py    MediaPipe Tasks extractor (pose + 2 hands) for images/videos/webcam
│
├── training/
│   ├── fetch_metadata.py         download + verify the official INCLUDE metadata
│   ├── fetch_dataset.py          optional raw-video/keypoint acquisition helpers
│   ├── audit_dataset.py          PHASE 2 -> docs/dataset_audit.md (+ results/dataset_audit_summary.json)
│   ├── calibrate_geometry.py     re-derive X_AXIS_SCALE from the release
│   ├── preprocess.py             PHASE 4 -> datasets/processed/sequences_<protocol>.npz
│   ├── models.py                 build_lstm / build_gru / build_gru_mha (+ parameter_count)
│   ├── train_common.py           seeding, data loading, callbacks, artefact writing; one code path
│   ├── train_lstm.py             thin wrappers around run_cli()
│   ├── train_gru.py
│   ├── train_gru_mha.py
│   ├── evaluate_models.py        PHASE 7 -> summary_metrics.csv, per_class_comparison.csv, plots
│   ├── robustness.py             PHASE 8 -> robustness_results.csv + degradation plots
│   └── tune_threshold.py         confidence-gate sweep -> threshold_sweep.csv
│
├── backend/inference/
│   ├── predictor.py              PredictionEngine + TemporalSmoother (gate, smoothing, caption, TTS hook)
│   ├── frames.py                 FrameBuffer, video/dataset -> features (same preprocessing as training)
│   ├── audio.py                  Speaker (gTTS / pyttsx3 / silent fallback)
│   ├── audit.py                  JSONL logging, feedback join, filters, metrics aggregation
│   ├── realtime.py               PHASE 12 webcam loop with HUD, keys, overrides
│   └── batch.py                  offline/batch inference over clips or video files
│
├── backend/
│   ├── main.py                   PHASE 10 FastAPI service
│   ├── schemas.py                Pydantic v2 request/response models (the trust boundary)
│   └── personalization.py        bounded profiles, calibration set, optional fine-tune
│
├── frontend/                    browser dashboard (index.html, app.js, app.css)
├── tests/                        PHASE 14 pytest suite (conftest + 8 modules)
├── scripts/download_models.py    MediaPipe .task bundles (+ checksums)
├── datasets/{raw,processed,metadata}/   raw clips local; metadata + shipped feature tensor versioned
├── models/{lstm,gru,gru_mha}/    model.keras, best.keras, class_mapping.json, preprocessing_config.json, training_history.json
├── results/                      evaluation, robustness, threshold artefacts (versioned deliverables)
├── logs/                         predictions.jsonl, feedback/feedback.jsonl, tts/
└── docs/                         audit, evaluation, model card, system card, limitations, setup, architecture
```

## 2. Data flow

```
INCLUDE keypoints (zip)                        webcam / uploaded video
        │                                              │
        │ backend/preprocessing/include50.read_keypoints_for   │ backend/preprocessing/mediapipe_extractor
        ▼                                              ▼
 (T, 75, 3) + (T, 75) mask  ────►  landmarks.pack_frame / normalise_sequence  ◄──── (75,3)+mask per frame
        │                                              │
        ▼                                              ▼
 sequence.resample_sequence → (30, 225)          FrameBuffer (ring buffer, 30 frames)
        │                                              │
        ▼                                              ▼
 datasets/processed/sequences_<protocol>.npz    PredictionEngine.predict(...)
        │                                              │
        ├── train_{lstm,gru,gru_mha}.py ──► models/    ├── realtime.py  (HUD + keys)
        │                                              ├── batch.py     (CSV output)
        └── evaluate_models.py / robustness.py         └── backend/main.py → frontend/index.html
                    │                                              │
                    ▼                                              ▼
                results/*.csv,*.png                    logs/*.jsonl → backend/inference/audit.py
```

**The live path and the training path share `backend/preprocessing/landmarks.py` and
`backend/preprocessing/sequence.py`.** A clip is turned into features by the same functions
whether it came from the dataset, a file upload or a webcam frame, so the model never
sees a different representation at inference time than it saw during training.

## 3. Key contracts

| contract | where | guarantee |
|---|---|---|
| landmark layout | `config.LandmarkLayout` | pose `[0,33)`, left hand `[33,54)`, right hand `[54,75)`; 225 features (300 with mask) |
| class order | `datasets/metadata/vocabulary.json` | one frozen order used by data, models, API and UI; each model directory stores a copy in `class_mapping.json` |
| sequence length | `config.SEQUENCE_LENGTH` (default 30) | nothing hard-codes 30: data builder, buffer and API validation all read the config |
| split protocol | `config.SPLIT_PROTOCOL` | `session_disjoint` (default, leak-free) or `official` (reported as leaking); both indices are cached as `datasets/metadata/index_<protocol>.csv` |
| prediction result | `backend.inference.predictor.PredictionResult` | sign, confidence, status, latency, model, raw argmax, stability, caption, timestamp |
| audit record | `backend/inference/audit.log_prediction` | `event_id, timestamp, model, predicted_class, confidence, status, latency_ms, fps, human_action, corrected_class, source, profile` |
| API prediction response | `backend.schemas.PredictionResponse` | `{sign, confidence, status, latency_ms, model, …}` — never a fabricated label; 503 when no weights exist |
| model artefacts | `training/train_common.save_artifacts` | `model.keras`, `best.keras`, `class_mapping.json`, `preprocessing_config.json`, `training_history.json` |

## 4. Configuration

Every tunable lives in `config.py` and can be overridden with an `ISL_`-prefixed
environment variable (or a `.env` file), for example:

```bash
ISL_SEQUENCE_LENGTH=40 ISL_CONFIDENCE_THRESHOLD=0.8 python run.py realtime
ISL_SPLIT_PROTOCOL=official python run.py preprocess
```

Groups: dataset paths & keypoint source · landmark layout & masking · axis
calibration · sequence handling & augmentation · splits & seeds · training
hyper-parameters · backend/inference/UX (gate, smoothing, stability, cooldown, TTS) ·
API/frontend (host, ports, upload limits, extension allow-lists) · personalisation
bounds · runtime (threads, verbosity).

## 5. Extension points

* **New architecture** — add a builder in `training/models.py`, a two-line
  `training/train_<name>.py` wrapper, and register it in `config.AVAILABLE_MODELS`;
  evaluation, robustness and inference pick it up automatically.
* **New features** — extend the layout in `config.LandmarkLayout` and
  `backend/preprocessing/landmarks.py`; `MODEL_INPUT_DIM` and the API validation follow.
* **New data** — implement a reader returning `(coords (T,75,3), mask (T,75), info)`
  (see `backend/preprocessing/include50.read_keypoints_for`) and reuse everything downstream.
* **Dashboard updates** — change the HTML, CSS or JavaScript under `frontend/`; the API remains the contract.

## 6. Design decisions worth knowing

* **No fabricated data anywhere.** Missing models → 503; missing dataset → loud
  failure naming the path; empty audit log → zeros in `/metrics`.
* **One inference code path.** `PredictionEngine` is used by the CLI, the API and the
  tests, so gate/smoothing behaviour cannot drift between them.
* **Simplicity over cleverness in metrics.** Precision/recall/F1 are computed in
  `training/evaluate_models.py` from an explicit confusion matrix (verified against
  hand-computed values in the tests), so the reported numbers do not depend on an
  opaque library call.
* **Everything measurable is logged.** Latency, FPS, confidence and human actions
  land in JSONL, which is what makes the error-review view and `/metrics` real.
