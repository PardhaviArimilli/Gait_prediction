# Checkpoint Status — Parkinson's FOG Project

## Completed
- Scaffolding & Configs
  - configs (data/training/model/postprocess/ensemble)
  - logging utility with rotation; logs/ folder and conventions
  - validation script (`src/data/validate.py`)
- Data utils
  - windowing (`src/data/windows.py`)
  - window dataset (`src/data/dataset.py`)
  - sampler utility (`src/train/sampler.py`)
- Evaluation
  - postprocess (thresholds, hysteresis, min-duration, merge)
  - scorer (IoU matching, AP/mAP)
- Model & Pipeline
  - `cnn_bilstm` model
  - pipeline to load a batch and forward pass
- OOF & Training
  - minimal OOF (`src/train/save_oof.py`) → `artifacts/oof/cnn_bilstm_oof.csv`
  - minimal training run (`src/train/run_minimal.py`) → checkpoint `artifacts/checkpoints/cnn_bilstm_min.pt`
  - subject GroupKFold splits for defog (`src/data/make_splits.py`) → `artifacts/splits/defog_folds.csv`
  - fold 0 OOF subset (`src/train/run_fold_oof.py`) → `artifacts/oof/cnn_bilstm_fold0.csv`
  - fold 0 training (`src/train/run_fold_train.py`) with window AP validation → best checkpoint `artifacts/checkpoints/cnn_bilstm_fold0_best.pt`
  - mixed training (defog + tdcsfog, 3-class) (`src/train/run_mixed_train.py`) — verified one short epoch
  - threshold tuning (fold 0 windows) (`src/eval/tune_thresholds.py`) → `artifacts/postprocess/thresholds_fold0.json`
  - isotonic calibration (fold 0 windows) (`src/eval/calibrate_isotonic.py`) → `artifacts/postprocess/iso_fold0_*.joblib`
  - fold report (fold 0) with calibration/thresholds (`src/eval/report_fold.py`) → `artifacts/reports/fold0_report.csv`
  - second model (TCN) OOF (fold 0) → `artifacts/oof/tcn_fold0.csv`
  - concatenated OOF and simple ensemble weights (`src/ensemble/concat_oof.py`, `src/ensemble/fit_weights.py`) → `artifacts/oof/oof_all.csv`, `artifacts/ensemble/weights.json`
  - ensemble fold report (fold 0) → `artifacts/oof/ensemble_fold0.csv`
  - full folds 0–4 trained for CNN-BiLSTM and TCN (no file caps) with OOF saved per fold/model; ensemble weights refit across folds 0–4 → `artifacts/ensemble/weights.json`
- Website (Netlify)
  - Modernized UI with navbar, slider; About page with PD/FOG info
  - Displays CV metrics (macro AP and per-model APs) via `metrics.json`

## In Progress / Next
- Proper training loop
  - Full sequence detection training (framewise logits) with losses (focal + temporal + boundary)
  - Early stopping and per-epoch validation mAP via scorer
  - Save best-by-mAP checkpoints per fold
- Full OOF generation
  - Train per fold with GroupKFold
  - Save OOF probabilities per fold and concatenate
- Threshold tuning & calibration
  - Optimize thresholds on val; isotonic/Platt calibration on OOF
  - Save postprocess params under `artifacts/postprocess/`
- Ensembling
  - Fit convex weights on OOF; save domain-aware weights
  - Optional stacker
- Cross-domain evaluation
  - Train tdcsfog ↔ test defog and reverse; report ΔmAP
- SSL pretraining (optional)
  - Contrastive/masked pretraining on unlabeled parquet; fine-tune
- Website (Netlify) for gait-parameter screening
  - Implement rules v1; optional tiny model v2

## Notes
- Determinism and logging are enabled; reference scorer implemented.
- Dataset loading currently filters to defog CSVs with labels (`StartHesitation`, `Turn`, `Walking`).
- Artifacts are stored under `artifacts/` with separate subfolders for splits, oof, and checkpoints.

## Cleanups
- Temporary inference tests added then removed after passing (2 tests).
