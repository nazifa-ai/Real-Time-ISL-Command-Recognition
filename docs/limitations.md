# Limitations

Every item below is a measured property of this project or of the INCLUDE-50
dataset, not a hypothetical. Where a number appears it was produced by
`training/audit_dataset.py` and is reproducible from `results/dataset_audit_summary.json`.

## 1. Scope

* This is an **isolated sign/command recognition** prototype. It classifies one sign at a
  time from a short clip. It is **not** a sign-language translator: it does not model ISL
  grammar, sentence structure, fingerspelling, non-manual markers (facial expression, mouth
  morphemes, head tilt) or spatial referencing.
* The system recognises exactly the classes present in the trained model's vocabulary
  (INCLUDE-50: 50 signs). Anything outside that vocabulary either produces a wrong label or
  should be caught by the confidence gate - it cannot produce a sensible "I don't know that
  sign" answer beyond the `UNCERTAIN` state.
* This is not a medical, legal, emergency or accessibility-certified system, and must not be
  used as the sole channel of communication.

## 2. Dataset limitations (measured)

| limitation | measurement |
|---|---|
| **Signer metadata absent** | INCLUDE publishes no signer ids, so signer-disjoint evaluation is impossible. Restated in `docs/dataset_audit.md` §8. |
| **Session leakage in the official split** | 199 of 299 recording clusters (bursts of consecutive file ids) span more than one side of the official train/test split; 63% of clips share an id-block with the other side. Benchmark numbers computed on the official split are therefore optimistic. |
| **Our mitigation** | The project's primary protocol (`session_disjoint`) keeps whole recording clusters on one side: 0 of 299 clusters span a split. Both numbers will be reported side by side. |
| **Very small test set per class** | The session-disjoint split yields 3-4 test clips per class (154 total). Per-class precision/recall/F1 therefore has wide confidence intervals and must not be read to three decimal places. |
| **One class has no test clip in the official split** | `DRY` appears only in the official train file; it is excluded from official-split per-class test metrics rather than silently dropped. |
| **Demographic and geographic bias** | INCLUDE was recorded by a small number of experienced signers in India, in a studio-like setting. ISL itself varies regionally. Performance for other signers, dialects, and skin tones is unknown and will very likely be lower. |
| **No signer-disjoint generalisation evidence** | Because signer ids are unavailable, no claim of signer-independent performance can be made. |
| **Released keypoints are 2-D + z** | x/y are image coordinates; z is MediaPipe's relative depth, which is far noisier. The 225-feature layout is preserved, but the effective spatial information is mostly 2-D. |
| **Axis anisotropy in the release** | The official keypoints were produced through a width/height-swapped resize, compressing x by ~3.16x. Measured and corrected (`config.X_AXIS_SCALE`); residual uncertainty is the spread between the two anatomical priors (3.00 and 3.26). Any result computed *without* the correction would train a squashed body model that does not transfer to the webcam. |
| **No FPS or duration metadata** | Not present in the keypoint release, so temporal statistics are reported in frames. |

## 3. Modelling limitations

* Landmark-based models discard appearance: they cannot use facial expression, finger
  contact with a surface, or any cue MediaPipe cannot see. Fast or partially occluded signs
  lose most of their signal.
* The three architectures (LSTM, GRU, GRU+attention) are all small recurrent models trained
  from scratch on ~650 clips. No pretrained backbone is used, so absolute accuracy is
  bounded by data size, not by architecture choice alone.
* Augmentation changes the landmark statistics on purpose. Aggressive settings can make the
  validation distribution differ from the test distribution; the augmentation policy must be
  identical across the three models or the comparison is void (enforced in
  `backend/preprocessing/sequence.augment`, called from one place).
* Robustness experiments perturbe landmarks directly. They measure sensitivity to *feature*
  degradation, which is a proxy for - not a substitute for - real occlusion, motion blur and
  bad lighting.

## 4. Deployment and human-factors limitations

* Webcam input is distributionally different from the studio recordings (camera height,
  framing, distance, lighting, background clutter). No cross-domain claim is made.
* Confidence is a softmax score, not a calibrated probability. It is used as a *gate*, and
  the threshold must be tuned on validation data (`training/tune_threshold.py`), never
  assumed optimal. The default 0.70 is a starting point, not a validated choice.
* Temporal smoothing introduces latency: a prediction needs `STABLE_FRAMES_REQUIRED`
  consistent frames before it is emitted.
* Text-to-speech output is English wording for ISL signs; it is a captioning aid, not a
  faithful translation of ISL grammar.
* Personalisation only adjusts thresholds/vocabulary and optionally fine-tunes on the user's
  own consented samples. It is deliberately conservative and cannot silently retrain itself
  from corrections.

## 5. Claims this project does not make

* It does not claim the attention model beats the GRU/LSTM baselines before the experiment
  has been run and reported.
* It does not claim the confidence threshold of 0.70 is optimal.
* It does not claim benchmark accuracy equals real-world accuracy.
* It does not claim to understand Indian Sign Language; it classifies isolated signs from a
  benchmark vocabulary.
