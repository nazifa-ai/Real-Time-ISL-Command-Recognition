# System card — real-time ISL command recognition service

This is the **system** view (what is deployed, what data flows where, who can
override what). The model itself is described in `docs/model_card.md`.

---

## 1. Purpose and scope

A command-recognition assistant: a user signs one of 50 INCLUDE-50 words; the system
shows the recognised sign, a confidence percentage, an `UNCERTAIN` state when it is
not confident, a growing caption, and optionally speaks the sign. A human can
override every decision. It is a **prototype**, not a translator and not a certified
accessibility device (`docs/limitations.md`).

## 2. Components

```
                    ┌──────────────────────────────┐
  webcam ──────────►│ backend/inference/realtime.py        │  local machine only
  (MediaPipe Tasks) │ buffer → gate → smoothing →  │  (cv2 window, keys a/c/r/s)
                    │ caption → TTS → JSONL log    │
                    └───────────────┬──────────────┘
                                    │
 video upload ──────────────────────┤
 dataset clip ──────────────────────┤      ┌────────────────────────┐
 JSON features ─────────────────────┘─────►│ backend/main.py        │ FastAPI
                                            │ /predict /feedback     │ Pydantic
                                            │ /personalize /metrics  │ validated
                                            │ /audit/events ...      │
                                            └───────────┬────────────┘
                                                        │ HTTP (JSON)
                                            ┌───────────▼────────────┐
                                            │ frontend/index.html + JS    │ browser dashboard
                                            │ live · upload · demo   │
                                            │ error review · metrics │
                                            │ personalisation        │
                                            └────────────────────────┘
```

| component | file | notes |
|---|---|---|
| shared prediction engine | `backend/inference/predictor.py` | one code path for gate, smoothing, stability, duplicate suppression, caption, TTS |
| frame → features | `backend/inference/frames.py` | webcam buffer, video file, dataset clip — all use the training preprocessing |
| audit log / error review | `backend/inference/audit.py` | append-only JSONL, filters, metrics aggregation |
| speech | `backend/inference/audio.py` | gTTS or pyttsx3; degrades to "unavailable" without crashing |
| API | `backend/main.py`, `backend/schemas.py` | no fabricated predictions; 503 + exact command when no weights exist |
| personalisation | `backend/personalization.py` | bounded, opt-in, stored as features |
| UI | `frontend/index.html`, `frontend/app.js`, `frontend/app.css` | browser client served by FastAPI |

## 3. Decision logic (identical in every entry point)

1. **Confidence gate** — `confidence >= ISL_CONFIDENCE_THRESHOLD` (default 0.70) else
   the label becomes `UNCERTAIN`. The threshold is tunable, per-profile (within
   ±0.2) and *not* claimed optimal (`docs/evaluation.md` §5).
2. **Temporal smoothing** — 3-frame sliding-window majority vote (min 2 votes).
3. **Stability** — a label must persist for 3 consecutive updates before it counts
   as `CONFIDENT`; until then the status is `PENDING`.
4. **Duplicate suppression + cooldown** — a held sign is not re-emitted (default
   3 s cooldown), so TTS does not repeat and the caption does not fill with copies.
5. **Speech** — only when confident **and** stable **and** new/cooldown-expired, and
   only if TTS is enabled and available.
6. **Human override** — ACCEPT / CORRECT / REJECT on any emission; stored with the
   prediction it refers to.

Statuses surfaced to the user: `CONFIDENT`, `PENDING`, `UNCERTAIN`, `IDLE`.

## 4. Data handling and privacy

| data | stored where | retention |
|---|---|---|
| uploaded video | temporary directory (`tempfile.mkdtemp`) | deleted immediately after feature extraction; copied to `datasets/raw/uploads/` **only** if `PERSIST_UPLOADS=true` |
| webcam frames | in-process ring buffer | never written to disk (only an optional annotated mp4 if `--record` is passed by the user) |
| prediction metadata | `logs/predictions.jsonl` | append-only; local file |
| human feedback | `logs/feedback/feedback.jsonl` | append-only; local file |
| personalisation | `models/personalized/<profile>/` | profile JSON + `calibration.npz` of **features** (no video) + optional fine-tuned weights |
| speech audio (gTTS) | `logs/tts/*.mp3` | local, only when a network TTS backend is used; pyttsx3 writes nothing |

No analytics, no telemetry, no third-party calls at prediction time (gTTS is the only
outbound call and only when speech is enabled). Secrets belong in `.env`
(`.env.example` documents every variable); `.gitignore` excludes `.env`, datasets,
model binaries, logs and all media.

## 5. Human oversight and personalisation policy

* Every emission can be overridden; the error-review view filters by low confidence,
  rejected, corrected, class, model and status, and exposes the predicted→corrected
  confusion table.
* Corrections **do not** retrain anything automatically. Adaptation is an explicit,
  separate step.
* Bounded personalisation: vocabulary can only be *restricted* to a subset of the 50
  official classes; the per-user threshold may move at most ±0.2 from the global
  default; fine-tuning requires ≥20 confirmed clips, runs 5 low-LR epochs on a
  **copy** of the base model and reports its calibration-set accuracy with an
  explicit caveat. Scientific results in `results/` are never affected by
  personalised weights.

## 6. Failure modes and expected behaviour

| failure | system behaviour |
|---|---|
| no trained model on disk | `/predict` → 503 with the exact `python training/train_<model>.py` command; `/health` → `degraded`, `model_loaded=false`; UI shows the not-reachable/not-trained notice |
| dataset/keypoints missing | audit/preprocess fail loudly with the path they looked for; API still serves `/health` |
| MediaPipe detects no person | buffer stores zeroed frames with mask 0; prediction becomes `UNCERTAIN` (measured: with both hands removed 142/154 test clips fall below the gate) |
| camera unavailable (server/Docker/WSL) | `backend/inference/camera.probe_cameras()` tests indices 0-3 and `realtime.py` prints the measured reason plus fallbacks instead of crashing; `python run.py camera` reports devices, indices and GUI support |
| no display available | the GUI is probed in a **subprocess** (a windowless `cv2.imshow` aborts the process through Qt/xcb), and real-time mode continues windowless with an explanatory notice; `--no-window` forces it |
| TTS engine missing or failing | speech silently disabled; captions keep working (unit-tested) |
| upload wrong type / too big / fake | 415 / 413 / 415 with a clear message, file never written outside a temp dir |
| API hammered | in-process rate limit (240 req/min/IP, `API_RATE_LIMIT_PER_MIN`) |
| log file corrupt line | that line is skipped, the rest of the audit log still loads |

## 7. Deployment notes

* `Dockerfile` and `docker-compose.yml` run one FastAPI service on port 8000. FastAPI
  also serves the browser dashboard. Compose mounts `models/`, `datasets/`, `results/`
  and `logs/` so model assets and audit data stay on the host.
* **Webcam limitation:** containers have no `/dev/video0` and a browser camera is
  owned by the host, so live capture inside Docker is not supported. Fallbacks:
  (a) run `backend/inference/realtime.py` on the host against the same code,
  (b) use the browser dashboard's upload/dataset tabs or live capture, which sends
  sampled frames from the browser to the API.
* CPU-only by design; measured latency is in `docs/evaluation.md` §7.
* Camera input needs a capture device *and* (for the preview window) a display; both are
  checked at start-up and reported with the exact fix (`docs/setup.md` §7).

## 8. Monitoring

`GET /metrics` reports, from the audit log: totals, status distribution,
`uncertain_rate`, confidence mean/min/max, latency mean/p50/p95/max, throughput,
human-action counts, corrections, human agreement rate and the most frequent
predicted classes. This is real logged traffic — an empty service reports zeros.
