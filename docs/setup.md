# Setup and operation

Tested on: **Python 3.13.14, Linux x86_64, CPU-only (2 vCPU, ~2 GB RAM)** with the
pinned versions in `requirements.txt`. TensorFlow 2.21 + Keras 3, MediaPipe 1.0.1
(Tasks API only — that release has **no** `mediapipe.solutions`).

---

## 1. Install

```bash
git clone <your-fork> isl-recognition && cd isl-recognition
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
python run.py doctor          # environment + asset self-check
```

Everything after this is driven by `python run.py <command>`; each command forwards
extra flags to the underlying script (`python run.py train --model gru --epochs 40`).

## 2. Get the data (required for audit/training)

The project trains on AI4Bharat's released **INCLUDE keypoint archive** (≈0.66 GB), not
on the 56 GB raw video release.

```bash
# 1) official metadata (a few hundred KB) -> datasets/metadata/
python run.py metadata

# 2) keypoints.  Download Pose_Signs.zip from the OpenHands/INCLUDE Zenodo record and
#    place the extracted archive where the loader looks, or point the env var at it:
#      https://zenodo.org/record/6674324/files/INCLUDE.zip      (contains Pose_Signs.zip)
export ISL_KEYPOINTS_PATH=/path/to/Pose_Signs_fresh.zip
python run.py info            # the "keypoint archive" line must say OK
```

The loader accepts either the `.zip` archive or an extracted `Pose_Signs/` folder and
records the resolved path in `docs/dataset_audit.md`.

MediaPipe Task bundles (needed only for webcam/video input, not for the dataset):

```bash
python scripts/download_models.py           # holistic + pose + hand, with checksums
```

## 3. Reproduce the study (the whole pipeline)

```bash
python run.py audit           # PHASE 2 -> docs/dataset_audit.md, results/dataset_audit_summary.json
python run.py preprocess      # PHASE 4 -> datasets/processed/sequences_session_disjoint.npz (~37 MB)
python run.py train           # PHASES 5-7 -> models/{lstm,gru,gru_mha}/  (~2-3 min/model on 2 vCPU)
python run.py evaluate        # PHASE 7 -> results/summary_metrics.csv, per_class_comparison.csv, plots
python run.py robustness      # PHASE 8 -> results/robustness_results.csv + degradation plot
python run.py threshold       #         -> results/threshold_sweep.csv + plot
```

Both split protocols are supported everywhere:

```bash
python run.py preprocess --protocol official
python run.py evaluate   --protocol official     # -> *_official.csv / *_official.png
python run.py robustness --protocol official
```

Environment overrides (all optional, `.env` is read automatically):

```bash
ISL_SEQUENCE_LENGTH=40 ISL_CONFIDENCE_THRESHOLD=0.8 ISL_SPLIT_PROTOCOL=official python run.py train
```

## 4. Run the system

```bash
# 1) API and browser dashboard (http://localhost:8000; API docs at /docs)
python run.py api
#    or: uvicorn backend.main:app --host 0.0.0.0 --port 8000

# 2) Live webcam recognition (local machine, camera required)
python run.py realtime --model gru_mha --threshold 0.70
#    keys: q quit | s speech | a accept | c correct | r reject | 1/2/3 model | x clear | b backspace

# 3) Offline / batch
python run.py batch --from-dataset --split test --limit 20
python run.py batch --video uploads/hello.mp4
python run.py batch --video-dir uploads/ --recursive
```

## 5. Tests

```bash
python run.py test                 # or: python -m pytest tests -v
python -m pytest tests/test_api.py -q
```

Tests that need the dataset or trained models **skip with an explicit reason** when
the artefact is absent, so the suite is meaningful on a fresh checkout and complete on
a full one. No test writes into the real logs (audit paths are redirected to a temp
dir) and no test writes into `models/personalized/`.

## 6. Docker

```bash
docker compose up --build          # API + browser dashboard :8000
docker compose run --rm api python training/audit_dataset.py
docker compose run --rm api python -m pytest tests -q
```

Mount `models/`, `datasets/`, `results/`, `logs/` (compose does this) so artefacts and
audit data live on the host.

### Webcam limitations in Docker

A container normally has no `/dev/video0`, and a browser camera belongs to the host
machine, so **local webcam recognition is not available inside Docker**. Supported
alternatives:

1. **Run `backend/inference/realtime.py` on the host** against the same checkout — full camera
   rate, the intended way to demo real-time behaviour.
2. **Use the browser dashboard** at `http://localhost:8000` for video upload,
   dataset demos or browser-camera capture. Browser capture and network transfer add
   latency compared with the local real-time camera loop.
3. **Run the API in Docker** for upload/dataset workflows. The Python real-time camera
   loop still runs on the host because it needs direct camera access and local MediaPipe.

## 7. Why the webcam does not work

Run the dedicated check first — it measures the two things a webcam needs (a capture
device and, for the preview window, a display) and prints the matching fix:

```bash
python run.py camera          # or: python scripts/camera_check.py --indices 0 1 2 3
```

Measured example from a headless container (this is what "not working" usually is):

```
platform            : Linux 6.1.158+
capture devices     : none (/dev/video* absent)
DISPLAY             : <unset>
OpenCV GUI backend  : Qt5
preview window      : False  (Available platform plugins are: xcb.)
  [--   ] index 0: opened=False readable=False ... device could not be opened
 * No /dev/video* device exists: this machine has no camera attached
 * No usable display: cv2.imshow here aborts the process -> use --no-window
```

| symptom | cause | fix |
|---|---|---|
| `no /dev/video*` / `device nodes: none` | the machine has no camera: server, container, WSL, CI, cloud sandbox, or VM without USB passthrough | use the browser dashboard upload/dataset tabs, `backend/inference/batch.py`, or run on a desktop machine |
| `opened=False` on every index although a camera is attached | camera already in use by another app (Zoom/Teams/browser tab) | close the other app, then re-run |
| same, on macOS | the *terminal application* was never granted camera access | System Settings → Privacy & Security → Camera → enable Terminal/iTerm/VS Code, then restart it |
| same, on Windows | camera access blocked for desktop apps | Settings → Privacy & security → Camera → allow desktop apps |
| index 0 fails, another index works | laptop has an internal + external camera, or a virtual camera is registered first | `python run.py realtime --camera 1` (the script already falls back through indices 0-3) |
| WSL2: no device at all | WSL does not forward USB cameras by default | forward it: `usbipd list` → `usbipd attach --wsl --busid <id>`, then check `/dev/video0` |
| Docker: no device | containers do not get `/dev/video0` automatically | `docker run --device /dev/video0 ...` plus `--group-add video`, and run with `--no-window` (no X server) — see §6 |
| process **dies with no Python traceback** after printing `qt.qpa.xcb: could not connect to display` | `cv2.imshow` with no display aborts the process in the C++ layer | this is handled now: `realtime.py` probes the GUI in a subprocess and continues windowless (`--no-window` forces it). If you see it, you are running an older copy |
| `cv2.error: The function is not implemented ...` on `imshow` | OpenCV was installed as `opencv-python-headless` (the pinned default, which has no GUI) | `pip install opencv-python` (non-headless) for a desktop preview window; headless is fine for uploads, batch and dataset work |
| camera works but every sign is `UNCERTAIN` | detector is fine, model confidence is low (that is the gate working) | check the hand-detection rates in the UI/HUD, hold the sign for the full buffer, and tune `--threshold` after reading `docs/evaluation.md` §5 |

Reminder: the hosted/sandbox preview you may be looking at has **no camera device and
no display**, so real-time video cannot work there by construction — that is why the
live tab buffers browser captures and why `realtime.py` prints fallbacks instead of a
traceback. Use `python run.py batch --from-dataset`, a video upload, or a desktop
machine for camera-rate recognition.

## 8. Troubleshooting

| symptom | cause / fix |
|---|---|
| `keypoint archive ... --` in `run.py info` | keypoints not found; set `ISL_KEYPOINTS_PATH` (see §2) |
| `FileNotFoundError: trained model not found` | train first: `python run.py train` (or `--model gru_mha`) |
| `/predict` returns 503 | same cause; `/health` reports `model_loaded=false` with the reason |
| MediaPipe prints `cuInit`/`oneDNN`/`absl` noise | harmless on CPU builds; the scripts filter it in logging |
| `Could not open camera index 0` | headless/container environment — see §6 and §7 |
| TTS says unavailable | no `gtts`/`pyttsx3` or no audio player; recognition still works (`TTS_ENGINE=none` disables it explicitly) |
| frontend shows "Backend not reachable" | start the API first (`python run.py api`) — the UI never fakes a prediction |
| Docker build is slow | TensorFlow is a large wheel; the layer cache makes rebuilds cheap |

## 9. Housekeeping

* `logs/predictions.jsonl` and `logs/feedback/feedback.jsonl` grow append-only; delete
  them to reset the error-review view (or use `backend.inference.audit.clear_logs()`).
* Evaluation results, trained model weights, the processed session-disjoint tensor
  and MediaPipe task bundles are kept in the project for a reproducible demo.
* Raw recordings and uploaded videos are not included in the repository; uploads are
  never retained unless `ISL_PERSIST_UPLOADS=true`.
