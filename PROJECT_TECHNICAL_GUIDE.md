## Parkinson's Freezing of Gait (FOG) — Technical Guide

This guide explains the end-to-end system in this repository and how to run training, evaluation, ensembling, inference, and the companion website. It aligns with the current code under `src/`, `configs/`, and `artifacts/`.

---

## High-level Overview

- **Goal**: Detect and score FOG-related events from wearable accelerometer data, and surface cross-validation metrics in a simple website for screening.
- **Core components**:
  - Data pipeline: validation, resampling, windowing, transforms, splits
  - Models: `cnn_bilstm`, `tcn` (framewise logits)
  - Training: fold-based CV, OOF writing
  - Evaluation: threshold tuning, isotonic calibration, CV metrics
  - Ensembling: convex weights fit on OOF
  - Serving: batch/ensemble inference and submission helpers
  - Website: Netlify-ready static site with metrics display and Care Finder

---

## Repository Structure (key paths)

```
configs/                  # YAML configs used across stages
  data.yaml               # paths, sample rate, logging, scorer IoU
  training_baseline.yaml  # trainer hyperparams/defaults
  model_cnn_bilstm.yaml   # model-specific config
  model_tcn.yaml          # model-specific config
  postprocess.yaml        # thresholds/hysteresis/durations, etc.
  ensemble.yaml           # ensemble weight search settings

src/
  common/                 # env + logging utilities
  data/                   # validate, windows, dataset, transforms, splits
  models/                 # cnn_bilstm, tcn, registry
  train/                  # training loops and orchestrators
  eval/                   # thresholds, calibration, scorer, reports, metrics
  ensemble/               # OOF concatenation + weight fitting
  serve/                  # inference (single/ensemble), CLI, submission utils

artifacts/                # checkpoints, oof, postprocess, ensemble, metrics
logs/                     # all run logs
website/netlify/          # static site (Home, About/Care)
```

See also: `IMPLEMENTATION_GUIDE.md` (detailed plan), `PROJECT_DESCRIPTION.md` (overview with diagrams).

---

## Environment and Setup

- Python ≥ 3.10 recommended.
- Install base dependencies:

```powershell
pip install -r requirements.txt
```

- Install PyTorch separately for your CUDA/CPU setup (see PyTorch website for the correct command). Add optional libs as needed.

Project expects the dataset layout already present under `train/defog`, `train/tdcsfog`, `test/defog`, `test/tdcsfog`, plus metadata files (`subjects.csv`, `defog_metadata.csv`, `tdcsfog_metadata.csv`, `events.csv`).

---

## Configuration (single source of truth)

Primary knobs live in `configs/`:

- `data.yaml`: paths, `sample_rate_hz` (100), `window_s` (5), `overlap` (0.5), logging, scorer IoU (0.5)
- `training_baseline.yaml`: batch size, optimizer, scheduler, early stop, seed
- `model_cnn_bilstm.yaml`, `model_tcn.yaml`: channels, layers, dilations, dropouts, etc.
- `postprocess.yaml`: threshold tuning and hysteresis/duration parameters
- `ensemble.yaml`: search space and constraints for convex weights

The code reads configs to avoid hardcoding paths or constants.

---

## Data Pipeline

Key modules:

- `src/data/validate.py`: schema/rate checks for raw CSVs; logs issues
- `src/data/windows.py`: windowing utilities (5 s, 50% overlap typical)
- `src/data/transforms.py`: standardization and simple augmentations
- `src/data/dataset.py`: `WindowDataset` yielding tensors and labels
- `src/data/make_splits.py`, `make_splits_tdcs.py`: subject-level CV splits

Typical flow:
1) Validate raw data (both sources). For tdcsfog, resample 128 → 100 Hz.
2) Window into 5 s segments with 50% overlap.
3) Apply transforms (e.g., per-window standardization; light jitter/scale/rotate in train only).
4) Use GroupKFold splits by subject for CV.

---

## Models

- `src/models/cnn_bilstm.py`: CNN backbone followed by BiLSTM; outputs framewise logits.
- `src/models/tcn.py`: Dilated temporal convolution network; outputs framewise logits.
- `src/models/registry.py`: Builder pattern to instantiate models by name.

Both produce `(batch, time, num_classes)` logits used for window AP and post-processing.

---

## Training and OOF Generation

Entrypoints in `src/train/`:

- `run_all_folds.py`: orchestrates all folds for the default/baseline run
- `run_fold_train.py`, `run_fold_train_tcn.py`: single-fold training for specific model
- `run_fold_oof.py`, `run_fold_oof_tcn.py`: write OOF for a fold using the best checkpoint
- `pipeline.py`, `loop.py`: shared training logic (optimizer, scheduler, early stop)

Example: train all folds for both models (adjust as needed):

```powershell
python -c "import sys, os; sys.path.append(os.getcwd()); from src.train.run_all_folds import main; main(folds_to_run=(0,1,2,3,4))"
```

Artifacts:
- Best checkpoints under `artifacts/checkpoints/*_fold{K}_best.pt`
- OOF predictions per fold under `artifacts/oof/...`
- Training logs in `logs/`

---

## Evaluation: Thresholds, Calibration, Metrics

Key modules in `src/eval/`:

- `tune_thresholds.py` (and `tune_thresholds_event.py`): per-class threshold search
- `calibrate_isotonic.py` (and `calibrate_isotonic_all.py`): isotonic calibration
- `window_ap.py`: AP computation over windows
- `scorer.py`: deterministic scoring utilities
- `compute_cv_metrics.py`: aggregates CV metrics and writes `artifacts/metrics/metrics.json`
- `report_fold.py`, `report_fold_ensemble.py`: fold-wise reports

Typical sequence after training/OOF:

```powershell
python -c "import sys, os; sys.path.append(os.getcwd()); from src.eval.tune_thresholds import main; main()"
python -c "import sys, os; sys.path.append(os.getcwd()); from src.eval.calibrate_isotonic import main; main()"
python -c "import sys, os; sys.path.append(os.getcwd()); from src.eval.compute_cv_metrics import main; main()"
```

Outputs:
- Post-process params in `artifacts/postprocess/`
- CV metrics in `artifacts/metrics/metrics.json` (copy to `website/netlify/metrics.json` to display on the site)

---

## Ensembling

OOF-based convex-weight ensembling is implemented in `src/ensemble/`:

- `concat_oof.py`: align and concatenate OOF predictions from multiple models
- `fit_weights.py`: fit non-negative weights summing to 1 to maximize AP

Command:

```powershell
python -c "import sys, os; sys.path.append(os.getcwd()); from src.ensemble.fit_weights import main; main()"
```

Artifacts:
- `artifacts/ensemble/weights.json`
- Logs in `logs/ensemble.log` and reports under `artifacts/reports/` as applicable

---

## Inference and Submission

Batch and ensemble inference under `src/serve/`:

- `infer.py`: run a single model with saved thresholds/calibrators
- `ensemble_infer.py`: run using learned ensemble weights
- `submission.py`: helpers to format outputs for submission
- `cli.py`, `summarize.py`: simple CLI and summarization tools

Example usage:

```python
from src.serve.infer import main as infer

# Configure paths/checkpoints/thresholds inside infer.py or via configs
infer()
```

Outputs:
- Per-file probabilities/intervals; summary CSVs in `artifacts/infer/...`

---

## Website (Netlify)

- Static site in `website/netlify/` with:
  - Inputs for gait parameters and rule-based screening
  - Metrics panel that reads `metrics.json` if present
  - Care page (About) and optional hospital pre-fetch JSON (`scripts/find_hospitals.py`)

To preview locally:

```powershell
python -m http.server 8888 -d website/netlify
```

To deploy, use Netlify with `netlify.toml` (publish directory `website/netlify`).

---

## Logging and Reproducibility

- Logs are written under `logs/` (training, eval, ensemble, inference).
- Seed and env controls live in `src/common/env.py`.
- Metrics and run settings should be captured in `artifacts/metrics/metrics.json` and related artifact folders.

---

## Quickstart

1) Install dependencies and ensure data is in place.
2) Generate subject-level folds (if not already present) via `src/data/make_splits*.py`.
3) Train folds and write OOF:

```powershell
python -c "import sys, os; sys.path.append(os.getcwd()); from src.train.run_all_folds import main; main(folds_to_run=(0,1,2,3,4))"
```

4) Tune thresholds, calibrate, compute metrics:

```powershell
python -c "import sys, os; sys.path.append(os.getcwd()); from src.eval.tune_thresholds import main; main()"
python -c "import sys, os; sys.path.append(os.getcwd()); from src.eval.calibrate_isotonic import main; main()"
python -c "import sys, os; sys.path.append(os.getcwd()); from src.eval.compute_cv_metrics import main; main()"
```

5) Fit ensemble and evaluate:

```powershell
python -c "import sys, os; sys.path.append(os.getcwd()); from src.ensemble.fit_weights import main; main()"
```

6) Run inference (single model or ensemble) and update website metrics.

---

## Common Issues and Troubleshooting

- Missing PyTorch: install the correct build for your CUDA/CPU from the official instructions.
- Path errors: verify `configs/data.yaml` paths and that artifacts directories exist; run from repo root so relative paths resolve.
- No OOF found: ensure you ran the fold OOF scripts (`run_fold_oof*.py`) or orchestrator that writes OOF after training.
- Metrics not visible on website: copy `artifacts/metrics/metrics.json` to `website/netlify/metrics.json`.
- Determinism: if results vary, set seeds in `src/common/env.py` and avoid nondeterministic CUDA ops.

---

## Notes

- `IMPLEMENTATION_GUIDE.md` contains a more opinionated, step-by-step plan with extensions (SSL, domain adapters). This technical guide stays aligned to the current code and artifacts present in the repo.


