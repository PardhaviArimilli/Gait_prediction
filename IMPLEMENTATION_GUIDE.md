# Implementation Guide — Parkinson's FOG Detection (Start → Finish)

This guide turns `design_doc_parkinsons_fog_enhanced.md` into an actionable, end-to-end plan. It emphasizes DRY, simplicity, and modularity so new models can be added with minimal changes.

## 0) Principles
- DRY: single sources of truth for data paths, sampling rate, scaler params, thresholds, scorer version, and logging.
- Simple first: working baseline > clever. Add complexity via optional flags.
- Modular: consistent interfaces for datasets, models, losses, post-processors, ensembling.
- Reproducible: fixed seeds; versions logged; deterministic scorer.

## 1) Repository Layout
```
configs/                # YAML configs per experiment/model
  data.yaml             # shared data & paths
  training_baseline.yaml
  model_cnn_bilstm.yaml
  postprocess.yaml
  ensemble.yaml
src/
  common/
    __init__.py
    env.py              # constants, paths, seeds, log levels
    logging_utils.py    # file + console loggers, rotation
    utils.py            # general helpers
  data/
    __init__.py
    schema.py           # dataclasses for samples, labels
    io.py               # CSV/Parquet I/O
    resample.py         # 128→100 Hz polyphase
    validate.py         # schema/rate/flatline checks
    features.py         # metadata joins, task ids
    windows.py          # windowing utilities
    dataset.py          # PyTorch Dataset(s) with domain/task conditioning
    scalers.py          # fit/transform, save/load
  models/
    __init__.py
    registry.py         # model factory registry
    gboost.py           # feature-based model (optional)
    cnn.py              # Conv1D blocks
    cnn_bilstm.py       # CNN-BiLSTM (+attention)
    tcn.py              # dilated TCN
    transformer.py      # encoder variant
  losses/
    __init__.py
    focal_bce.py
    temporal_consistency.py
    boundary_loss.py
  train/
    pipeline.py         # end-to-end train per fold (loads configs)
    loop.py             # train/val loops
    sampler.py          # positive-aware & hard-negative sampling
    optimizer.py
    scheduler.py
    checkpoints.py
  eval/
    scorer.py           # deterministic IoU≥0.5, PR/mAP
    metrics.py
    postprocess.py      # thresholds, hysteresis, NMS
    calibrate.py        # Platt/Isotonic
    analyze.py          # diagnostics/plots
  ensemble/
    weights.py          # convex weight fit on OOF
    stacker.py          # meta-learner (optional)
  serve/
    infer.py            # batch/online inference with context buffer
website/
  netlify/              # minimal site for gait parameters
logs/
README.md
IMPLEMENTATION_GUIDE.md
```

## 2) Environment & Tooling
1. Python ≥ 3.10; create venv.
2. Core libs: numpy, pandas, scipy, pyarrow, scikit-learn, lightgbm, xgboost (optional), pytorch+cuda, torchaudio (optional), tqdm, pyyaml, mlflow or wandb.
3. Dev: black, isort, flake8/ruff, mypy (optional).
4. Set `PYTHONHASHSEED`, `CUBLAS_WORKSPACE_CONFIG=:16:8`, `CUDNN_DETERMINISTIC=1` for determinism.

## 3) Configuration (Single Source of Truth)
- `configs/data.yaml` (example):
```yaml
paths:
  root: .
  raw_train_defog: train/defog
  raw_train_tdcs: train/tdcsfog
  raw_unlabeled: unlabeled
  metadata:
    subjects: subjects.csv
    defog: defog_metadata.csv
    tdcs: tdcsfog_metadata.csv
    events: events.csv
sample_rate_hz: 100
window_s: 5
overlap: 0.5
normalization: source_global_then_subject_residual
logging:
  dir: logs
  level: INFO
scorer:
  iou: 0.5
```
- Model/training configs inherit base settings; only override differences.

## 4) Data Pipeline
A. Validation & Catalog
- Run `src/data/validate.py` over raw folders: check columns, monotonic time, sample rate, no flatlines; log to `logs/pipeline.log`.
- Build a catalog (CSV/Parquet) of all recordings with subject_id, source, visit, medication, task availability, sample count.

B. Resampling & Standardization
- tdcsfog: polyphase resample 128→100 Hz with anti-alias; defog: passthrough.
- Persist resampling method/params as sidecars.
- Fit per-source global scalers; optionally subject residual scalers; save to `artifacts/scalers/`.

C. Metadata & Task Conditioning
- Join `subjects.csv`, `defog_metadata.csv`, `tdcsfog_metadata.csv`.
- From `events.csv`, derive task ids; if tasks not available at inference, plan a light task classifier later.

D. Windowing
- For window models: 5 s windows, 50% overlap; mark windows overlapping positives; emit dataset splits by subject.
- For detectors: prepare variable-length sequences (30–60 s) with overlap.

## 5) Split Strategy
- GroupKFold by subject_id (k=5). Persist fold assignment in `artifacts/splits/fold_{k}.csv`.
- Keep a private holdout (10–15% subjects) not touched until final.

## 6) Datasets & Samplers
- `src/data/dataset.py` yields tensors (T×C), labels (framewise), and optional task/domain embeddings.
- `src/train/sampler.py` implements positive-aware sampling (30–50% positive windows) and hard-negative quotas (turning w/o FOG).

## 7) Models (Registry & Modularity)
- `src/models/registry.py` provides `build_model(name:str, **kwargs)`.
- Implement minimal set first:
  - `cnn.py`: 4 Conv1D blocks with mixed kernels/dilations.
  - `cnn_bilstm.py`: Conv → BiLSTM(128) → attention → logits.
  - Add `tcn.py`, `transformer.py` as optional follow-ups.
- Each model exposes a uniform forward: `(batch, time, channels) → framewise logits (batch, time, num_classes)`.

## 8) Losses
- Combine via weights from config:
  - `focal_bce` (γ∈[1,2])
  - `temporal_consistency` on logits differences (λ≈0.1)
  - `boundary_loss` with Gaussian emphasis at starts/ends (λ≈0.05)

## 9) Training Loop (Per Fold)
- `src/train/pipeline.py --config configs/training_baseline.yaml`:
  1) Load data config, fold split, and model config.
  2) Construct datasets/dataloaders with samplers.
  3) Build model from registry; create optimizer (AdamW) and scheduler (cosine + warmup).
  4) Train epochs with early stopping on val mAP; log to `logs/train_fold_{k}.log` and tracker (W&B/MLflow).
  5) Save best checkpoint and OOF predictions for this fold to `artifacts/oof/model_name/fold_{k}.parquet`.

## 10) Post‑processing & Calibration
- Use `src/eval/postprocess.py` to apply per-class thresholds + hysteresis + duration prior + NMS.
- Tune thresholds on val to maximize mAP; calibrate probabilities with isotonic/Platt on OOF.
- Persist thresholds/calibration params to `artifacts/postprocess/model_name/`.

## 11) Deterministic Scorer
- `src/eval/scorer.py` implements:
  - Convert framewise probs→intervals (using saved thresholds) per class.
  - Matching rule (IoU≥0.5), tie-breaking by score.
  - Compute PR and AP with monotonic interpolation; output mAP per class + macro.
- Provide unit tests and golden test vectors.

## 12) Evaluation & Diagnostics
- Run scorer on each fold’s val set, produce:
  - mAP (class/macro), event-level PR, onset latency, false alarms/hour.
  - Domain-wise breakdown (tdcsfog vs defog), confusion/time analyses.
- Save reports to `artifacts/reports/fold_{k}/` and log summaries in `logs/eval_fold_{k}.log`.

## 13) Ensembling
- Collect OOF predictions from top N models.
- Level 1: average or rank-average seeds/folds of same model.
- Level 2: `src/ensemble/weights.py` fits convex weights on combined OOF to maximize mAP (constraints w≥0, Σw=1). Save weights per domain (lab/home) if domain-aware.
- Optional stacker: `src/ensemble/stacker.py` (logistic/MLP) trained strictly on OOF.
- Save final ensemble config and weights to `artifacts/ensemble/` and evaluation to `logs/ensemble.log`.

## 14) Cross‑Domain Validation
- Train on tdcsfog → test on defog, and reverse; report ΔmAP. Use domain adapters (BN/GRL/adapters) as ablations.

## 15) SSL Pretraining (Optional but Recommended)
- `src/models/ssl/` (if added later) trains with contrastive/masked objectives on unlabeled:
  - crops 10–30 s, batch target 256–512 (grad accumulation ok), 50–100 epochs.
  - augment ranges as per design doc.
- Save pretrained weights; re-run training with `pretrained_backbone=true`.

## 16) Inference (Batch/Online)
- `src/serve/infer.py`:
  - Load checkpoint, scalers, thresholds.
  - Batch: predict per file; Online: rolling buffer (10–30 s), stride 1–2 s.
  - Output intervals per class and summary CSV.
- Log latency & errors to `logs/inference.log`.

## 17) Website (Netlify) — Gait Parameter Screening
- `website/netlify/` modern static app:
  - Horizontal navbar with links to Home and About pages.
  - Inputs: cadence, stride length, step time variability, symmetry, turn duration, age group.
  - Slider for cadence alongside numeric entry for quick exploration.
  - Rules v1 (thresholds) in a single `rules.json`.
  - Displays CV metrics (macro AP and per-model APs) from `metrics.json` if present.
  - Deploy via `netlify.toml` (publish `website/netlify`).

## 18) Logging & Debugging
- Default handlers write to `logs/` with rotation (10 MB × 10 files) and console at INFO.
- Every run header: run id, git commit, seed, config hashes, subject groups, domain mix.

## 19) Reproducibility & Versioning
- Pin env in `requirements.txt` + `requirements-lock.txt`.
- Track artifacts with MLflow/W&B; record scorer version and thresholds alongside checkpoints.

## 20) Milestone Checklist
- Week 1: pipeline + baseline CNN window model + CV working + OOF cache.
- Week 2: CNN‑BiLSTM/TCN detectors + post‑processing sweep + domain adapters + start SSL.
- Week 3: Transformer + temporal/boundary losses + cross‑domain + finish SSL.
- Week 4: Ensembling + calibration + clinical metrics + ablations + finalize reports.

## 21) Add a New Model (Modularity Recipe)
1. Create `src/models/my_model.py` implementing the standard forward interface.
2. Register in `src/models/registry.py`.
3. Add config `configs/model_my_model.yaml` with hyperparams.
4. Train via the same `src/train/pipeline.py` (no other code changes).
5. OOF predictions plug into ensembling automatically.

## 22) DRY Guardrails
- Shared config keys validated at startup; warn on duplicates.
- Paths, sample rate, scorer IoU, thresholds live only in configs/artifacts; code reads, never hardcodes.
- Single scorer implementation imported everywhere.

## 23) Go/No-Go Readiness
- Passing data validation; deterministic scorer tests; baseline mAP reproduced across machines; logs present with rotation.

---
Run order (typical):
1. Validate and catalog → resample/standardize → fit scalers → window/sequence datasets.
2. Generate folds → train per fold → save best + OOF.
3. Tune post‑process thresholds + calibrate → evaluate with scorer.
4. Fit ensemble weights → evaluate.
5. Cross‑domain tests → finalize.
6. Optional: SSL pretrain → fine‑tune → refresh steps 2–4.
7. Package inference and website.
