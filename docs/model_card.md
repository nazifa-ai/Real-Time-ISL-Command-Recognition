# Model card — ISL command recognition (INCLUDE-50)

Following the model-card convention (Mitchell et al., 2019), adapted to a
final-year project: what the models are, what they were trained on, how they
perform, and where they must not be trusted.

---

## 1. Model details

| | LSTM baseline | GRU baseline | **GRU + Multi-Head Attention (proposed)** |
|---|---|---|---|
| checkpoint | `models/lstm/model.keras` | `models/gru/model.keras` | `models/gru_mha/model.keras` |
| architecture | Input → LSTM(128) → LSTM(64) → Dense(128) → Softmax(50) | Input → GRU(128) → GRU(64) → Dense(128) → Softmax(50) | Input → GRU(128, return_sequences) → GRU(64, return_sequences) → MultiHeadAttention(4 heads, key/value dim 64, residual + LayerNorm) → temporal pooling (average ⊕ max) → Dense(128) → Softmax(50) |
| parameters | 245,426 | 188,338 | 263,026 |
| input | `(None, None, 225)` (variable T; training uses 30) | same | same |
| output | `(None, 50)` softmax, `datasets/metadata/vocabulary.json` order | same | same |
| framework | TensorFlow 2.21 / Keras 3, CPU-trained | same | same |
| training | Adam, lr 1e-3 with ReduceLROnPlateau, batch 32, dropout 0.3, label smoothing 0.05, early stopping (patience 12) | same | same |
| epochs run | 34 (best epoch 22) | 44 (best epoch 32) | 31 (best epoch 19) |
| seed | 42 (numpy, python, TensorFlow, offline augmentation) | same | same |

### Input features (shared by all three models)

* MediaPipe landmarks: **33 pose + 21 left hand + 21 right hand = 75 landmarks**
* per landmark: (x, y, z) → **225 features per frame**; a presence mask is available
  (`MASK_ENABLED`, off in this build: 225 → 300 features)
* normalisation is **per frame, relative to body geometry**: subtract the
  mid-shoulder point, divide by shoulder width (fallbacks: hip width, then centroid);
  missing landmarks become exact zeros, never NaN
* sequence: fixed **30 frames**, resampled (linear) or edge-padded from the clip;
  clips with fewer than 5 usable frames are rejected
* axis calibration: INCLUDE's released keypoints are anisotropic (x compressed by
  3.16× relative to y); both training and inference use the same `X_AXIS_SCALE`
  correction so there is no train/deploy mismatch (derivation in `docs/dataset_audit.md`)

---

## 2. Intended use

* **Primary use:** isolated **command** recognition — a user signs one of the 50
  INCLUDE-50 words, the system produces the label, a confidence value, an
  `UNCERTAIN` state when the gate is not met, and optionally speaks it.
* **Users:** the project's author and evaluators; a demonstrator for an assistive
  command vocabulary.
* **Interaction model:** the system is **advisory**. A human can ACCEPT, CORRECT or
  REJECT every emission; corrections are logged and reviewable.
* Personalisation is opt-in and deliberately bounded (vocabulary restriction,
  threshold within ±0.2, optional fine-tune on the user's own confirmed clips).

## 3. Out of scope / not to be used for

* **Not a translator.** No grammar, no sentence formation, no continuous signing.
* **Not "all of ISL".** 50 words from one dataset release, recorded under one
  collection protocol. Indian Sign Language has substantial regional and individual
  variation that this data does not represent.
* **Not a safety- or medical-critical system.** Do not use it for emergency,
  medical, legal or accessibility-certified communication.
* **No deaf-community validation.** This is a technical prototype; it was not
  co-designed with or evaluated by deaf signers. Fluency and acceptability were
  never assessed.
* Not for surveillance, identification of individuals, or any inference about a
  person beyond the 50 trained words.

## 4. Training data

* **AI4Bharat INCLUDE** (Sridhar et al., *INCLUDE: A Large Scale Dataset for Indian
  Sign Language Recognition*, ACM MM 2020, DOI 10.1145/3394171.3413528), CC-BY-4.0
  metadata, subset **INCLUDE-50** (50 words), official pre-extracted MediaPipe
  keypoints (4284 files in the release; 958 belong to the 50-word subset).
* Measured on this machine (`docs/dataset_audit.md`): 958 videos, 958/958 usable,
  0 corrupt, 14–25 usable clips per class (mean 19.16, sd 3.08), all 1080×1920,
  mean 63.6 frames per clip (median 60, max 154), pose detected 100% of frames,
  left hand 85.7%, right hand 79.3%, both hands 70.9%.
* Signer metadata: **not available in the dataset** — INCLUDE publishes no signer
  ids, so signer-disjoint training/evaluation is impossible. Recording-session
  clustering (file-id bursts) is used instead.
* No external data, no pretrained weights, no synthetic training data.
* Splits: `session_disjoint` (primary, leak-free by recording cluster) and
  `official` (AI4Bharat's files, reported as leaking). See `docs/evaluation.md` §1.

## 5. Evaluation

Primary protocol (`session_disjoint`, 154 held-out clips):

| metric | LSTM | GRU | **GRU+MHA** |
|---|---|---|---|
| accuracy | 0.6623 | 0.6299 | **0.7727** |
| macro F1 | 0.5969 | 0.5588 | **0.7182** |
| weighted F1 | 0.6025 | 0.5643 | **0.7255** |
| model-only latency (ms, batch 1) | 2.76 | 2.43 | 2.73 |

The current checked-in evaluation bundle contains the session-disjoint protocol only.
No official-split comparison or signer-independent accuracy claim is made here.

Per-class results, confusion matrices and failure analysis are in
`results/per_class_comparison.csv` and `docs/evaluation.md` §4. The CSV is the
authoritative source for the current per-sign results.

Robustness (`results/robustness_results.csv`): GRU+MHA falls by about 0.026 accuracy
when 40% of frames are dropped; removing both hands drops it to 0.117. These are
perturbations of extracted landmarks, not tests under changed image lighting.

## 6. Quantitative analysis caveats

* Test sets are small (154 / 192 clips; 3–4 clips per class), so per-class numbers
  are coarse and confidence intervals are wide. No statistical significance test is
  claimed: the proposed model's aggregate advantage (≈2.6 points of accuracy over the
  GRU) is within the noise that this test-set size can produce, and the honest
  reading is "attention does not hurt and is best on the aggregate metrics", not
  "attention is proven better".
* The `official` protocol numbers must never be quoted as generalisation.
* Latency figures are machine-specific (2 vCPU CPU-only sandbox).

## 7. Ethical, privacy and fairness considerations

* **Privacy by default:** the live system logs prediction metadata only; uploaded
  videos are processed in a temporary directory and deleted (`PERSIST_UPLOADS=false`).
  Personalisation stores *features*, not video.
* **Consent:** feedback/personalisation data is user-provided and opt-in; the audit
  log is local JSONL.
* **Representation risk:** the 50-word vocabulary and the recording protocol come
  from one dataset; performance on other signers, dialects, ages, skin tones,
  clothing, lighting and camera setups is unmeasured. The system should not be
  presented as broadly accurate.
* **Overclaiming risk:** the biggest ethical hazard for a project like this is
  over-promising a "sign language translator". The README, this card, the system card
  and `docs/limitations.md` state the scope in the first paragraph, and the UI shows
  `UNCERTAIN` rather than guessing.

## 8. Reproduction

```bash
python run.py doctor                 # environment + assets
python run.py audit                  # dataset audit  -> docs/dataset_audit.md
python run.py preprocess             # tensors       -> datasets/processed/sequences_<protocol>.npz
python run.py train                  # models        -> models/{lstm,gru,gru_mha}/
python run.py evaluate                # results/summary_metrics.csv, per_class_comparison.csv, plots
python run.py robustness              # results/robustness_results.csv
python run.py threshold               # results/threshold_sweep.csv
```

## 9. Citation

> Sridhar, A., Ganesan, R. G., Kumar, P., Khapra, M. (2020). *INCLUDE: A Large Scale
> Dataset for Indian Sign Language Recognition.* Proceedings of the 28th ACM
> International Conference on Multimedia, 4518–4525. DOI 10.1145/3394171.3413528

Keypoint extraction uses MediaPipe Hands/Pose (Google) via the Tasks API.
