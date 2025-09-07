# Overview
Goal: build robust, clinically useful models to detect Freezing of Gait (FOG) from wearable accelerometer data, explicitly addressing domain shift (lab vs home), label sparsity, and long‑range temporal dependencies. This expands the original plan with domain adaptation, self‑supervised pretraining, task conditioning, temporal‑aware objectives, and clinical evaluation.

# Objectives & Success Criteria
- **Primary metric:** mean average precision (mAP) across FOG event types (Start Hesitation, Turning, Walking). Report per‑class AP and macro mAP.
- **Targets:**
  - Baseline single best model mAP ≥ 0.30
  - Ensemble mAP ≥ 0.38
  - ≥ 15% fewer false positives vs best single model at matched recall
- **Clinical secondary:** sensitivity at 95% specificity; detection latency (onset delay) ≤ 0.5 s median.
- **Operational:** deterministic training, versioned data lineage, reproducible CV and OOF artifacts.

# Data
- **Sources:**
  - `tdcsfog` — lab‑controlled, 128 Hz, 833 recordings (cleaner, structured tasks)
  - `defog` — at‑home, ~100 Hz, 91 recordings (noisier, real‑world)
  - `unlabeled` — 67 large parquet files (tens of millions of samples) for self‑supervised pretraining
- **Signals:** 3‑axis accelerometer: vertical (V), mediolateral (ML), anteroposterior (AP).
- **Labels:** Event intervals for three FOG classes: StartHesitation, Turning, Walking.
- **Metadata:** `subjects.csv`, `defog_metadata.csv`, `tdcsfog_metadata.csv`, `events.csv`, `daily_metadata.csv`.

## Data Access & Storage
- Keep raw CSV/Parquet immutable.
- Derive unified Parquet per recording: columns `[t, v, ml, ap, subject_id, source, medication_state, task_type, visit_number]`.
- Store labels as interval tables: `[start_s, end_s, class, task_context]`.
- Persist per‑recording JSON sidecars for sample rate, resampling method, filter flags, scaler params.
- Version datasets and features with DVC or Git‑LFS.

## Preprocessing
- **Resampling (unify to 100 Hz):**
  - tdcsfog: polyphase downsample 128 → 100 Hz
  - defog: passthrough if ~100 Hz; minor jitter correction if needed
- **Sync & trimming:** align time bases; trim leading/trailing NaNs; ensure monotonic `t`.
- **Calibration/units:** normalize per source to zero mean, unit variance; optionally convert from g to m/s²; persist scalers.
- **Filtering (ablation):** 0.5–20 Hz 4th‑order Butterworth band‑pass; keep a no‑filter variant.
- **Windowing:**
  - Classification heads: sliding windows of 5 s, 50% overlap; also ablate 3/7 s.
  - Detection heads: variable‑length sequences (30–120 s) with overlapping context.
- **Standardization:** evaluate per‑subject vs global z‑score; prefer per‑source global + subject residual normalization.
- **Augmentations:** jitter (Gaussian), small 3D rotations, time‑warp, magnitude scaling, random crop/pad; MixUp/CutMix for window models.
- **Task conditioning:** join `events.csv` to annotate task segments and add categorical task ids; derive task embeddings at train time.

## Splits & Leakage Control
- **CV protocol:** GroupKFold by `subject_id` (e.g., 5 folds). No subject appears in both train and validation.
- **Holdout:** 10–15% subjects as private test for final selection.
- **Stratification:** maintain class‑wise positive duration across folds; ensure both sources (tdcsfog/defog) appear in train/val for each fold.

# Problem Framing
- **Event detection (preferred):** framewise logits → post‑process to intervals (hysteresis, duration priors, NMS).
- **Window classification (supporting):** per‑window multi‑label probabilities → stitch via thresholding + hysteresis.
- **Task‑aware modeling:** include task embeddings (e.g., TUG, Turning, MB tasks, 4MW) to modulate priors and thresholds.

# Model Zoo (Enhanced)
- **A. Feature‑based Gradient Boosting (GBoost)**
  - Time, frequency, wavelet, cross‑axis, gait proxies; plus metadata features (medication on/off, visit number, age bin, disease duration, UPDRS, NFOGQ, source domain, task id).
  - LightGBM or XGBoost, one‑vs‑rest with calibrated probabilities.
- **B. 1D CNN (local patterns)**
  - 4–6 Conv1D blocks (mixed kernels 5/9/15, dilations), BN, ReLU, Dropout.
  - (B1) Window classifier; (B2) sequence detector with anchor‑free framewise logits.
- **C. CNN‑BiLSTM + Attention (local + temporal)**
  - Shallow Conv → BiLSTM(128–256) → self‑attention → temporal pooling.
- **D. Temporal Convolutional Network (TCN)**
  - Dilated causal/residual stacks; add attention block on top for long context.
- **E. Transformer Encoder (long‑range)**
  - 4–6 layers, d_model=256, 4–8 heads, relative positional encodings (T5/DeBERTa‑style); performer/linformer if needed.
- **F. SSL‑pretrained backbone**
  - Pretrain CNN/TCN/Transformer on unlabeled with contrastive or masked modeling; fine‑tune for detection.
- **Domain adaptation hooks (for B–F):**
  - Source‑specific BatchNorm, domain adversarial head (GRL), or feature‑wise affine adapters; optional domain tokens for transformers.

# Losses & Labeling
- **Primary:** framewise BCEWithLogits or focal loss (γ ∈ [1, 2]) for class imbalance.
- **Temporal consistency loss:** encourage smoothness between adjacent frames (Huber on first differences of logits).
- **Boundary‑aware auxiliary:** increase confidence near annotated starts/ends (Gaussian weighting), and duration regularization to penalize unrealistically short blips.
- **Hard negative mining:** emphasize turning‑without‑FOG and gait transitions.

# Training Details
- **Optimizer:** AdamW. lr=1e‑3 (CNN/TCN), lr=3e‑4 (Transformer). Cosine decay + 5% warmup.
- **Batching:** by time budget (e.g., 64×5 s windows or 30–60 s sequences). Gradient accumulation for long contexts.
- **Regularization:** dropout 0.2–0.5, weight decay 1e‑4, mixup (window models), stochastic depth (transformer).
- **Early stopping:** patience 25 epochs on val mAP.
- **Mixed precision:** enable.
- **Checkpoints:** best‑by‑mAP and last; save OOF predictions per fold.

# Post‑processing
- **Thresholds:** per‑class tuned on CV; calibrate (Platt/Isotonic) on OOF.
- **Hysteresis:** start threshold high, continue threshold lower to reduce flicker.
- **Duration priors:** drop < 0.3 s; merge intra‑class gaps < 0.5 s.
- **NMS:** IoU=0.5 over intervals to suppress overlaps.
- **Task‑aware tweaks:** lower thresholds around tasks with higher prior FOG risk (e.g., turning), raise in rest.

# Ensembling
- **Level 1 (within‑model):** seeds, folds, minor hyperparams → average or rank‑average.
- **Level 2 (cross‑model):**
  - Weighted averaging with weights ≥ 0, sum=1, fit on OOF to maximize mAP.
  - Stacking: logistic regression or shallow MLP on OOF framewise scores + task/domain features.
  - Rule blend: favor detector models near turning/starts; favor GBoost in steady walking.
- **Domain‑aware weighting:** fit separate weights for tdcsfog and defog OOF; interpolate by a domain classifier.

# Evaluation Protocol
- **Primary:** mAP per class and macro from interval predictions.
- **Event‑level:** precision/recall/F1; onset detection latency.
- **Operational:** false alarms per hour; stability (prediction flicker rate).
- **Domain transfer:** train on one source, test on the other; report drop.
- **Clinical:** sensitivity at fixed specificity (e.g., 95%); PPV/NPV; per‑subject calibration (reliability curves).
- **Reporting:** mean ± std across folds; PR curves; calibration plots; ablations.

## Metric & Scorer Specification (deterministic)
- Framewise scores converted to intervals via class‑specific thresholds and hysteresis.
- A predicted interval matches a GT interval if IoU ≥ 0.5 in time; ties resolved by highest score.
- mAP computed from precision‑recall curves using monotonic interpolation; class‑wise and macro.
- Boundary tolerance during training: label smoothing around starts/ends (±0.1–0.2 s) to mitigate small offsets.
- Provide a reference scorer with fixed versions and unit tests.

# Self‑Supervised Pretraining (SSL)
- **Objectives:**
  - Temporal contrastive learning (SimCLR/TS2Vec‑style) with augmentations (time‑warp, jitter, rotations, masking).
  - Masked sequence modeling (MAE‑style) for 1D signals: reconstruct masked spans.
  - Cross‑axis agreement: encourage consistent embeddings across (V, ML, AP) permutations.
- **Data:** unlabeled parquet files (chunk to 10–30 s; random crops).
- **Transfer:** init backbones from SSL, then fine‑tune detection heads; freeze early layers for regularization sweep.

## SSL Practical Curriculum
- Batch target: 256–512 crops (with grad accumulation if needed), 50–100 epochs.
- Aug ranges: rotation ≤ 8°, time‑warp 0.9–1.1, jitter σ=0.01–0.03 g, masking 10–20% spans.
- Pretrain per‑domain (lab/home) and mixed; compare fine‑tune vs partial freeze.

# Experiment Tracking & Reproducibility
- Use W&B/MLflow; log configs, metrics, artifacts, OOF caches, and scalers.
- Set seeds, pin library versions, store preprocess configs; export data cards per dataset.

## Logging & Debugging
- Central `logs/` directory (git‑tracked with rotation) for pipeline, training, evaluation, and serving.
- Log levels: DEBUG (dev only), INFO, WARNING, ERROR. Default INFO for training; DEBUG in local dev.
- Structure (examples):
  - `logs/pipeline.log` — data ingestion, resampling, validation summaries, counts.
  - `logs/train_fold_{k}.log` — per‑fold training, metrics, loss curves, LR, model hash, seed.
  - `logs/eval_fold_{k}.log` — per‑fold evaluation, mAP breakdown, thresholds, calibration details.
  - `logs/ensemble.log` — weight search/stacking outputs, final weights, validation mAP.
  - `logs/inference.log` — batch/online inference stats, latency, errors.
- Rotation policy: 10 MB/file, keep last 10; archive older to `logs/archive/`.
- Include run id, subject ids used, domain mix, and git commit in every log header.

# Compute Plan
- **Hardware:** 1× 12–24 GB GPU suffices for CNN/TCN/Hybrid; Transformers may need gradient checkpointing.
- **Runtime:** 0.5–2 h for CNN/TCN/Hybrid; 2–4 h for Transformer depending on context.
- **Storage:** 50–100 GB for intermediates, checkpoints, OOF.

# Risks & Mitigations
- **Class imbalance / label sparsity:** focal loss, positive sampling, hard negatives, duration priors, SSL pretraining.
- **Domain shift (lab ↔ home):** per‑source normalization, domain adapters/GRL, domain‑aware ensembling; evaluate cross‑domain.
- **Subject leakage / overfitting:** GroupKFold by subject; holdout subjects; strong regularization.
- **Label noise / boundary uncertainty:** tolerance windows in scoring; boundary‑aware losses; post‑processing smoothing.
- **Data quality variance (home):** band‑pass + quality scoring; robust augmentations; threshold calibration per domain.

## Label Taxonomy & Harmonization
- defog: framewise `StartHesitation`, `Turn`, `Walking` indicate FOG event frames for respective classes; background otherwise.
- tdcsfog: `Event` + `Task` + metadata used to map to classes. If only "any FOG" is available, use as positive for a binary auxiliary head and rely on defog for class specificity; or map via aligned `events.csv` where applicable.
- Boundary policy: pad GT by ±N frames for tolerance in training losses; keep scorer strict (IoU ≥ 0.5).

## Task Conditioning Availability
- If `Task` is unavailable at inference, train a lightweight task classifier to produce task embeddings online; otherwise fall back to domain conditioning only.

## Temporal/Bndry Loss Weights
- Default: `L_total = L_focal + 0.1·L_temporal + 0.05·L_boundary`. Tune per class/domain.

## Resampling Integrity
- 128→100 Hz with anti‑alias filter; strict stride checks; boundary timestamp rounding documented and consistent.

# Milestones
1. **Week 1:** data pipeline with metadata integration; unified resampling; GroupKFold; baseline GBoost and CNN (window) + first OOF.
2. **Week 2:** CNN‑BiLSTM+Attention and TCN detectors; post‑processing sweep; domain adapters; start SSL pretraining.
3. **Week 3:** Transformer encoder with relative positions; temporal consistency/boundary losses; cross‑domain validation; finish SSL.
4. **Week 4:** ensembling (domain‑aware weights + stacking); calibration; clinical metrics; ablations; finalize deliverables.

## Domain Adaptation Ablations
- Compare None vs source‑specific BN vs GRL (λ schedule) vs small affine adapters; early stopping on domain‑balanced mAP.

## Operational Details
- Reproducibility: cuDNN determinism flags, seeded data loaders, pinned versions, hashed data cards; store scorer+threshold configs with artifacts.
- Data validation: schema/rate checks, flatline/saturation detectors, gravity orientation sanity; quality score used in sampling.
- Batching/imbalance: 30–50% windows overlapping positives; quotas for hard negatives (turning w/o FOG).
- Inference: 1–2 s stride, 10–30 s context buffer; CPU/GPU latency budget; fallback behavior if `Valid` missing.

## Website: Gait Parameter Screening (Netlify)
- Goal: a lightweight web tool where a user inputs gait parameters to receive “normal vs abnormal” screening (not a medical diagnosis).
- Hosting: Netlify (static front‑end + optional serverless function for simple rules or model inference).
- Inputs (initial set): cadence (steps/min), stride length (m), step time variability (%CV), step symmetry (%), turn duration (s), UPDRS‑like mobility score (optional), age group.
- Logic v1 (rules): simple threshold/range checks derived from literature/validation set; show contributing factors and confidence note.
- Logic v2 (optional): small exported classifier (e.g., LightGBM) trained on derived gait features from labeled windows; compiled to JS via `m2cgen` or served via Netlify function.
- UX: responsive form, instant feedback, data privacy note, “not for diagnosis” disclaimer, link to model card.
- CI: build previews, environment vars for thresholds, versioned via tags aligned with model releases.


# Deliverables
- Training repo with `configs/` per model and dataset variant.
- Preprocessing CLI scripts and data cards; persisted scalers.
- Best checkpoints and final ensemble; OOF predictions.
- Evaluation notebooks: CV reports, PR curves, latency, calibration, domain transfer.
- Model card: assumptions, limitations, safety notes.

# Appendix A — Feature List (expanded)
- **Time domain (per axis + magnitude):** mean, median, std, var, RMS, MAD, skew, kurtosis, range, IQR, ZCR, Hjorth (activity, mobility, complexity).
- **Frequency domain:** PSD band powers (0.5–3, 3–8, 8–12, 12–20 Hz), spectral centroid/spread/entropy/flatness, dominant freq/amplitude, harmonic ratio.
- **Cross‑axis:** corr(v, ml), corr(v, ap), corr(ml, ap), energy ratios, tilt statistics.
- **Wavelet:** db4/db6 level energies (L1–L5), wavelet entropy.
- **Gait cues:** cadence, stride interval variability, step regularity index.
- **Metadata:** medication state, visit, age bin, disease duration, UPDRS, NFOGQ, source domain, task id/embedding.

# Appendix B — Config Sketch (example)
```yaml
model:
  type: cnn_bilstm_attention
  conv_blocks: 3
  lstm_hidden: 128
  attention: true
  dropout: 0.3
  domain_adapter: bn_affine
train:
  lr: 0.001
  batch_size: 64
  epochs: 120
  loss: focal_bce + temporal_consistency
  optimizer: adamw
  warmup_pct: 0.05
  weight_decay: 1e-4
  early_stop_patience: 25
data:
  sample_rate: 100
  window_s: 5
  overlap: 0.5
  normalization: source_global_then_subject_residual
  augmentations: [jitter, rotate, timewarp]
  task_conditioning: true
postprocess:
  hysteresis: {start: 0.6, continue: 0.4}
  min_duration_s: 0.3
  merge_gap_s: 0.5
ensemble:
  domain_aware_weights: true
evaluation:
  metrics: [map_macro, map_per_class, latency_ms, false_alarms_per_hour]
  clinical: [sens_at_95_spec]
```

# Appendix C — Ensemble Weight Fitting (domain‑aware)
- Optimize `mAP(Σᵢ wᵢ pᵢ)` on OOF with `wᵢ ≥ 0`, `Σᵢ wᵢ = 1`.
- Fit weights separately on tdcsfog and defog OOF; at inference, blend weights using a small domain classifier score (prob(home)).


