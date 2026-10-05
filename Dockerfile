# ISL command recognition service (FastAPI + browser dashboard + CLI tools).
#
# What works inside a container: dataset audit, training, evaluation, robustness,
# the FastAPI service, browser dashboard, and batch inference over video files.
#
# What does NOT work inside a container by default: a *local* webcam
# (`backend/inference/realtime.py`) - containers usually have no /dev/video0.
# The browser dashboard uses browser camera access and sends sampled frames to
# the API. See README's webcam limitations for the supported local workflow.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    OPENCV_VIDEOIO_PRIORITY_LIST=FFMPEG \
    ISL_PERSIST_UPLOADS=false

# OpenCV headless needs these shared libraries; ffmpeg decodes uploaded videos.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libglib2.0-0 libsm6 libxext6 libxrender1 \
        curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

# The project includes the model weights and MediaPipe task bundles for the demo.
# Fetch bundles only if a task asset is missing.
RUN python scripts/download_models.py --check || python scripts/download_models.py

# Never run as root.
RUN useradd --create-home --uid 10001 isl && chown -R isl:isl /app
USER isl

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

# The FastAPI service also serves the browser dashboard.
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
