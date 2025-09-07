# Overview
Goal: build reliable models to detect Freezing of Gait (FOG) events from wearable accelerometer data, then combine them to improve accuracy and robustness. Scope covers data prep, model zoo, training plan, evaluation, ensembling, and delivery.

# Objectives & Success Criteria
- **Primary metric:** mean average precision (mAP) across event types (Start Hesitation, Turning, Walking). Report class-wise AP and macro mAP.
- **Targets:**
  - Baseline mAP ≥ 0.30
  - Ensemble mAP ≥ 0.38
  - False positive rate reduced ≥ 15% vs best single model at same recall.
- **Operational:** deterministic training scripts, clear data lineage, and reproducible results.

# Data
- **Sources:** `tdcsfog` (lab, 128 Hz), `defog` (at-home, ~100 Hz), optional `daily` (unlabeled, for pretraining/unsupervised only).
- **Signals:** 3-axis accelerometer: vertical (V), mediolateral (ML), anteroposterior (AP).
- **Labels:** event intervals for three FOG types.

## Data Access & Storage
- Keep raw zips immutable.
- Derive Parquet files per recording: columns `[t, v, ml, ap, subject_id, source]`.
- Store labels as interval tables `[start_s, end_s, class]` per file.
- Version with DVC or Git-LFS.

## Preprocessing
- **Resampling:** unify to 100 Hz. For 128 Hz, use polyphase resampling; for 100 Hz, passthrough.
- **Sync & trimming:** align streams to common time base; drop leading/trailing NaNs.
- **Calibration/units:** convert `g` to m/s² where needed or normalize per-source to zero mean, unit variance.
- **Filtering:** optional 0.5–20 Hz band-pass (Butterworth 4th) to remove drift and high-frequency noise. Keep a no-filter variant for ablation.
- **Windowing:** sliding windows of 5 s with 50% overlap for classification heads; for detection heads, keep sequence-level and predict event intervals.
- **Standardization:** per-subject or global z-score (decide via CV). Persist scaler params.
- **Augmentation:**
  - Jitter (Gaussian noise), small rotations of the 3D vector, time-warping, magnitude scaling, random cropping/padding.
  - MixUp/CutMix on windows for classification variants.

## Splits & Leakage Control
- **CV protocol:** GroupKFold by `subject_id` (e.g., 5 folds). No subject appears in both train and val.
- **Holdout:** final 10–15% subjects as a private test (never touched until final selection).
- **Stratification:** maintain class balance of positive intervals across folds.

# Problem Framing
We’ll try both **event detection** and **window classification**:
- **Detection (preferred):** predict start/end and class of FOG intervals. Post-process with non‑max suppression (NMS) and duration constraints.
- **Window classification (supporting):** per-window probability per class, then stitch into intervals via thresholding + hysteresis.

# Model Zoo
We’ll build complementary models that fail differently.

## A. Feature‑based Gradient Boosting (GBoost)
- **Features per window (examples):**
  - Time: mean, std, RMS, skew, kurtosis, zero‑crossing rate.
  - Frequency (FFT over 5 s): band power (0.5–3, 3–8, 8–12, 12–20 Hz), spectral centroid, spectral entropy, dominant freq + amplitude.
  - Cross‑axis: correlation coefficients, energy ratios, tilt angle stats.
  - Wavelets: discrete wavelet energies (db4/db6) by level.
  - Gait proxies: step cadence estimate (peak picking), stride regularity.
- **Model:** LightGBM or XGBoost, multi-label (one-vs-rest) with calibrated probabilities.
- **Why:** strong on tabular summaries; fast iterate; interpretable features.

## B. 1D CNN (Local patterns)
- **Backbone:** 4–6 conv blocks (Conv1D → BN → ReLU → Dropout), kernel sizes [5, 9, 15] mixed, dilations for context.
- **Head options:**
  - (B1) Per-window classifier.
  - (B2) Sequence detector using anchor-free framewise logits + NMS.

## C. CNN‑BiLSTM Hybrid (Local + temporal)
- **Backbone:** shallow Conv1D → BiLSTM(128–256) → temporal pooling.
- **Head:** framewise logits with CRF-like smoothing (post-process) or simple sigmoid + hysteresis.

## D. Temporal Convolutional Network (TCN)
- **Backbone:** dilated causal conv stacks (residual blocks) to capture long context efficiently.
- **Head:** framewise detection.

## E. Transformer Encoder (Long‑range)
- **Backbone:** 4–6 encoder layers, model dim 256, 4–8 heads, relative positional encodings.
- **Tricks:** performer/linformer variant if memory is tight; masking for variable-length.
- **Head:** framewise logits + duration prior.

# Losses, Class Imbalance, and Labels
- **Framewise BCE with logits** for multi-label outputs.
- **Focal loss** variant to focus on rare positives.
- **Positive sampling:** oversample windows overlapping labeled events; hard negative mining on confusing segments (turning without FOG).
- **Label smoothing:** small (ε = 0.05) to reduce overconfidence.

# Training Details
- **Optimizer:** AdamW. Start lr = 1e‑3 for CNN/TCN, 3e‑4 for Transformer. Cosine decay + warmup 5% steps.
- **Batching:** batch by total time (e.g., 64 windows of 5 s, or 30‑60 s sequences for detectors).
- **Regularization:** dropout 0.2–0.5, weight decay 1e‑4, mixup for window models.
- **Early stopping:** patience 25 epochs on val mAP.
- **Mixed precision:** enable for speed.
- **Checkpoints:** best-by-mAP and last.

# Post‑processing
- **Thresholding:** per-class thresholds tuned on CV (optimize mAP on val). Consider Platt scaling/Isotonic for calibration.
- **Hysteresis:** two-threshold scheme (high to start, low to continue) to reduce flicker.
- **Duration priors:** drop blips < 0.3 s; merge gaps < 0.5 s within same class.
- **NMS:** for overlapping detections, keep higher score; IoU threshold 0.5.

# Ensembling
- **Level 1 (within‑model):**
  - Different seeds, fold models, and minor hyperparams → average or rank-average probabilities.
- **Level 2 (cross‑model):**
  - **Weighted averaging:** optimize weights on out‑of‑fold (OOF) predictions with a simple constrained optimizer (weights ≥ 0, sum 1).
  - **Stacking:** train a meta‑learner (logistic regression or shallow MLP) on OOF framewise/window probabilities to predict final probs.
  - **Rule blend:** if models disagree, favor detector models near turning/starts; favor GBoost in steady walking.
- **Calibration after ensemble:** isotonic on OOF.

# Evaluation Protocol
- **Primary:** mAP by class and macro. Compute from interval predictions vs ground truth.
- **Secondary:** event‑level precision/recall, F1, and time‑to‑detection latency.
- **Ablations:**
  - Filtering vs none.
  - Window length 3/5/7 s.
  - Per‑subject vs global normalization.
  - With/without augmentations.
- **Reporting:** mean ± std across folds; learning curves; reliability plots.

# Experiment Tracking & Reproducibility
- **Tracking:** Weights & Biases or MLflow for params, metrics, artifacts, and OOF caches.
- **Determinism:** set seeds; log library versions; pin package hashes; save preprocess scalers.
- **Data cards:** YAML per dataset with resampling method, filters, and feature set.

# Compute Plan
- **Hardware:** 1× mid-range GPU (e.g., 12–24 GB VRAM) is enough. CPU-only for GBoost.
- **Time budget per run:** 0.5–2 h for CNN/TCN/Hybrid, 2–4 h for Transformer depending on context length.
- **Storage:** ~50–100 GB for intermediates, checkpoints, and OOF predictions.

# Risks & Mitigations
- **Class imbalance / label sparsity:** focal loss, positive sampling, augmentations.
- **Domain shift (lab vs home):** per-source normalization; domain‑adversarial loss (optional); source‑specific batch norm.
- **Overfitting to subjects:** GroupKFold; holdout subjects; strong regularization.
- **Noisy labels:** tolerance windows in scoring; soft labels around boundaries; robust post‑processing.

# Milestones
1. **Week 1:** data pipeline + baseline GBoost and 1D CNN; CV working; first OOF cache.
2. **Week 2:** CNN‑BiLSTM and TCN; post‑processing + threshold sweep; improved mAP.
3. **Week 3:** Transformer; finalize ensembling (weighted avg + stacking); calibration.
4. **Week 4:** hardening, ablations, documentation, model card, final holdout eval.

# Deliverables
- Training repo with `configs/` for each model and dataset variant.
- Preprocessing scripts with CLI.
- Saved checkpoints for top single models and final ensemble.
- Evaluation notebooks: CV reports, PR curves, confusion matrices, and calibration plots.
- Model card: assumptions, limitations, and safety notes.

# Appendix A — Feature List (starter)
- **Time domain (per axis + magnitude):** mean, median, std, var, RMS, MAD, skew, kurtosis, range, IQR, ZCR, Hjorth parameters (activity, mobility, complexity).
- **Freq domain (per axis + magnitude):** PSD band powers (0.5–3, 3–8, 8–12, 12–20 Hz), spectral centroid/spread/entropy/flatness, peak freq/amplitude, harmonic ratio.
- **Cross‑axis:** corr(v, ml), corr(v, ap), corr(ml, ap), energy ratios, tilt angle mean/std.
- **Wavelet:** level energies (db4/db6) L1–L5, wavelet entropy.
- **Gait cues:** cadence estimate, stride interval variability, step regularity index.

# Appendix B — Config Sketches
```yaml
# example: cnn_bilstm.yaml
model:
  type: cnn_bilstm
  conv_blocks: 2
  lstm_hidden: 128
  dropout: 0.3
train:
  lr: 0.001
  batch_size: 64
  epochs: 120
  loss: focal_bce
  optimizer: adamw
  warmup_pct: 0.05
  weight_decay: 1e-4
  early_stop_patience: 25
data:
  sample_rate: 100
  window_s: 5
  overlap: 0.5
  normalization: global_z
  augmentations: [jitter, rotate, timewarp]
postprocess:
  hysteresis: {start: 0.6, continue: 0.4}
  min_duration_s: 0.3
  merge_gap_s: 0.5
```

# Appendix C — Ensemble Weight Fitting
- Use OOF predictions to solve: minimize `1 - mAP(Σᵢ wᵢ pᵢ)` subject to `wᵢ ≥ 0`, `Σᵢ wᵢ = 1`.
- Practical: coordinate descent over simplex + random restarts; keep it simple and fast.

