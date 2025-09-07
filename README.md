## Parkinson’s Freezing of Gait (FoG) — Repository README

This repository contains a full ML/DL pipeline for detecting FoG‑related events from wearable IMU time series. It includes data windowing, model training (CNN‑BiLSTM, TCN), evaluation/metrics, ensembling, post‑processing to intervals, inference, and simple website assets.

For a deeper, concept‑focused walkthrough, see `ALGO.md`.

### What’s inside
- Data pipeline: `src/data/` (windowing, transforms), `configs/data.yaml`
- Models: `src/models/` (`cnn_bilstm.py`, `tcn.py`, `registry.py`)
- Training: `src/train/` (fold training, mixed training), `configs/training_baseline.yaml`
- Evaluation & metrics: `src/eval/` (window AP, interval scorer, CV metrics)
- Ensembling: `src/ensemble/` (OOF concat, weight fitting)
- Serving: `src/serve/` (single model and ensemble inference)
- Artifacts: `artifacts/` (checkpoints, metrics, ensemble weights, postprocess)
- Website assets: `website/netlify/`

### Requirements
- Python 3.9+ (3.10 recommended)
- PyTorch, NumPy, Pandas, scikit‑learn, joblib, PyYAML

Install
```powershell
pip install -r requirements.txt
```

### Data layout
- Labeled training: `train/defog/*.csv`, `train/tdcsfog/*.csv`
- Test input examples: `test/defog/*.csv`, `test/tdcsfog/*.csv`
- Metadata: `subjects.csv`, `defog_metadata.csv`, `tdcsfog_metadata.csv`, `events.csv`
- Unlabeled pool (optional): `unlabeled/*.parquet`

Key data parameters in `configs/data.yaml`:
- `sample_rate_hz: 100`, `window_s: 5`, `overlap: 0.5`

### Quick start — Train a fold
```powershell
python -c "import sys, os; sys.path.append(os.getcwd()); from src.train.run_fold_train import main; main(fold=0, epochs=100)"
```

Train TCN variant
```powershell
python -c "import sys, os; sys.path.append(os.getcwd()); from src.train.run_fold_train_tcn import main; main(fold=0, epochs=100)"
```

### Threshold tuning, calibration, and CV metrics
```powershell
python -c "import sys, os; sys.path.append(os.getcwd()); from src.eval.tune_thresholds import main; main()"
python -c "import sys, os; sys.path.append(os.getcwd()); from src.eval.calibrate_isotonic import main; main()"
python -c "import sys, os; sys.path.append(os.getcwd()); from src.eval.compute_cv_metrics import main; main()"
```
- Outputs:
  - Per‑fold thresholds/calibrators: `artifacts/postprocess/`
  - Cross‑validation metrics JSON: `artifacts/metrics/metrics.json` (also copied to `website/netlify/metrics.json`)

### Ensembling
Fit convex weights on OOF predictions to maximize AP:
```powershell
python -c "import sys, os; sys.path.append(os.getcwd()); from src.ensemble.fit_weights import main; main()"
```
Result: `artifacts/ensemble/weights.json`

### Inference
Single model inference over a directory of CSVs:
```powershell
python -m src.serve.infer --input_dir test/defog --output_dir artifacts/infer/defog_cnn \
  --checkpoint artifacts/checkpoints/cnn_bilstm_fold0_best.pt --model cnn_bilstm
```

Ensemble inference with post‑processing to intervals:
```python
from src.serve.ensemble_infer import ensemble_and_postprocess

ensemble_and_postprocess(
    input_dir="test/defog",
    out_root="artifacts/infer/ens_defog",
    ckpt_cnn="artifacts/checkpoints/cnn_bilstm_fold*_best.pt",
    ckpt_tcn="artifacts/checkpoints/tcn_fold*_best.pt",
)
```

### Export to ONNX (for web)
```powershell
python scripts/export_onnx.py --ckpt artifacts/checkpoints/tcn_fold0_best.pt --model tcn \
  --out website/netlify/assets/tcn.onnx --time_len 500
```

### Configs overview
- `configs/data.yaml`: paths, sample rate, windowing, overlap
- `configs/model_cnn_bilstm.yaml`, `configs/model_tcn.yaml`: model hyperparameters
- `configs/training_baseline.yaml`: epochs, optimizer, scheduler, focal gamma, sampler hints
- `configs/postprocess.yaml`: thresholds, hysteresis, min duration, merge gap (domain‑aware)
- `configs/ensemble.yaml`: averaging/weighting strategy and constraints

### Useful paths
- Checkpoints: `artifacts/checkpoints/*.pt`
- Ensemble weights: `artifacts/ensemble/weights.json`
- Metrics: `artifacts/metrics/metrics.json`
- Logs: `logs/*.log`

### Troubleshooting
- If imports fail in one‑liners, ensure the repo root is the current working directory.
- For CPU‑only environments, models run in eval mode; ensure checkpoints exist at referenced paths.
- Low AP? See the improvement plan in `ALGO.md` (thresholds/calibration, sampler, augmentations, architecture tweaks, smoothing).

### See also
- Detailed algorithm guide: `ALGO.md`
- Technical design notes: `PROJECT_TECHNICAL_GUIDE.md`, `IMPLEMENTATION_GUIDE.md`


