# Evaluation artefacts

This folder documents **how** the project is evaluated; the measured outputs live in
`results/` (so that they stay next to the plots and CSVs the app serves).

## Protocol in one page

1. **Split first, everything else second.** The split is fixed and cached before any
   model sees data (`datasets/metadata/index_<protocol>.csv`).
   * `session_disjoint` (primary): pools the full INCLUDE-50 pool, allocates whole
     recording clusters to one split → 652 train / 152 val / **154 test**, 0 of 299
     clusters spanning a split, all 50 classes in every split.
   * `official`: AI4Bharat's own train/test files + a seeded, class-stratified,
     session-aware validation carve-out → 686 / 80 / 192; **199 of 299 clusters span
     the split**, so it is reported as leaking.
2. **One preprocessing for everyone.** `training/preprocess.py` writes
   `sequences_<protocol>.npz`; the three training scripts read the same file.
3. **Identical training conditions.** Same seed, augmentation, callbacks, epochs,
   batch size, loss, label smoothing. Each run writes
   `class_mapping.json`, `preprocessing_config.json`, `training_history.json`.
4. **Frozen evaluation.** `training/evaluate_models.py` scores `X_test` with no
   test-time augmentation and computes accuracy, macro/weighted P/R/F1, per-class
   metrics, confusion matrices, parameter counts and single-clip latency.
5. **Perturbation study.** `training/robustness.py` applies 14 conditions to the same
   frozen tensors and pushes them through the real models.
6. **Gate study.** `training/tune_threshold.py` sweeps the confidence threshold and
   reports coverage vs selective accuracy (the shipped 0.70 is a starting point, not a
   proven optimum).

## Where each number comes from

| question | artefact | command |
|---|---|---|
| how good is each model on the leak-free split? | `results/summary_metrics.csv`, `results/per_class_comparison.csv`, `results/confusion_*.png`, `results/model_comparison.png` | `python training/evaluate_models.py` |
| how much does the official split inflate? | `results/summary_metrics_official.csv`, `results/evaluation_summary_official.json` | `python training/evaluate_models.py --protocol official` |
| which classes fail, and how? | `results/per_class_comparison.csv`, `evaluation_summary_<protocol>.json` (top confusions, weakest/strongest classes) | same as above |
| how fast is inference? | `results/latency_comparison.png`, `latency_ms_*`, `throughput_fps` in `summary_metrics.csv` | same as above |
| what breaks the models? | `results/robustness_results.csv`, `results/robustness_degradation.png` | `python training/robustness.py` (both protocols) |
| what does the confidence gate cost? | `results/threshold_sweep.csv`, `results/threshold_sweep.png` | `python training/tune_threshold.py` |
| how live behaviour is verified | `tests/test_inference.py` (gate, smoothing, stability, duplicate suppression, TTS-failure safety) | `python -m pytest tests/test_inference.py -v` |

## Reporting rules used in this project

* Every number is produced by a run in this repository; nothing is copied from the
  literature or estimated. Where the original paper reports ≈94.5% for INCLUDE-50, this
  README states only the difference between the two protocols *measured here*.
* Baseline and proposed model are named explicitly; the comparison table is the
  headline result.
* Failures are reported as prominently as successes (13 of 50 classes have zero recall
  on the leak-free split; that list is in `docs/evaluation.md` §4).
* No claim is made about live-camera accuracy, unseen signers or continuous signing —
  those were not measured (`docs/evaluation.md` §8).
