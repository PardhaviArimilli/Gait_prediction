import os
import json
import glob
# import joblib  # calibration disabled for stable metrics
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader  # not used after refactor, kept for compatibility

from src.data.dataset import WindowDataset
from src.models.registry import build_model
from src.eval.postprocess import probs_to_intervals
from src.eval.scorer import score_intervals, mean_ap
from src.common.logging_utils import get_logger
import yaml


LABELS = ["StartHesitation", "Turn", "Walking"]
FPS = 100.0

def load_fold_thresholds(fold: int, domain: str = "defog") -> dict:
    """Load domain thresholds from configs/postprocess.yaml for stable evaluation."""
    cfg_path = "configs/postprocess.yaml"
    if os.path.exists(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        # Prefer domain-specific thresholds;
        dom = (cfg.get("domains") or {}).get(domain) or {}
        thr = (dom.get("thresholds") or cfg.get("defaults", {}).get("thresholds") or cfg.get("thresholds") or {})
        # Map any key inconsistencies (e.g., Turning->Turn)
        mapped = {}
        for c in LABELS:
            if c in thr:
                mapped[c] = float(thr[c])
            elif c == "Turn" and "Turning" in thr:
                mapped[c] = float(thr["Turning"])
        if mapped:
            return mapped
    return {c: 0.5 for c in LABELS}

def load_fold_calibrators(fold: int):
    """Calibration disabled in metrics to avoid instability."""
    return None


def compute_fold_ap(model_name: str, fold: int) -> float:
    folds = pd.read_csv("artifacts/splits/defog_folds.csv")
    val_paths = folds[folds["fold"] == fold]["path"].tolist()
    if not val_paths:
        return 0.0
    window_s = 5
    overlap = 0.5
    ds = WindowDataset(val_paths, window_s=window_s, overlap=overlap, sample_rate_hz=FPS, label_cols=LABELS)

    model = build_model(model_name, num_classes=len(LABELS))
    ckpt = f"artifacts/checkpoints/{'cnn_bilstm' if model_name=='cnn_bilstm' else 'tcn'}_fold{fold}_best.pt"
    if os.path.exists(ckpt):
        state = torch.load(ckpt, map_location="cpu")
        try:
            model.load_state_dict(state, strict=False)
        except Exception:
            if isinstance(state, dict) and "state_dict" in state:
                model.load_state_dict(state["state_dict"], strict=False)
    model.eval()

    # Build per-window intervals (approximate event-level using window spans)
    thresholds = load_fold_thresholds(fold, domain="defog")
    calibrators = load_fold_calibrators(fold)
    preds_by_class = {c: [] for c in LABELS}
    gts_by_class = {c: [] for c in LABELS}
    step = int(window_s * FPS)
    hop = int(step * (1.0 - overlap))
    with torch.no_grad():
        for k in range(len(ds)):
            x, y = ds[k]
            fi, wi = ds.index[k]  # file index, window index
            x = x.unsqueeze(0)  # (1, T, C)
            logits = model(x)   # (1, T, K)
            pw = torch.sigmoid(logits).squeeze(0).cpu().numpy()  # (T, K)
            # per-window score by max over time
            scores = pw.max(axis=0)
            # calibration per class if available
            if calibrators:
                for ci, c in enumerate(LABELS):
                    if c in calibrators:
                        scores[ci] = float(calibrators[c].transform([scores[ci]])[0])
            # window time span
            start_s = (wi * hop) / FPS
            end_s = start_s + (step / FPS)
            for ci, c in enumerate(LABELS):
                if scores[ci] >= thresholds.get(c, 0.5):
                    preds_by_class[c].append((start_s, end_s, float(scores[ci])))
                if y.numel() and y[ci].item() >= 0.5:
                    gts_by_class[c].append((start_s, end_s))

    ap_by_class = score_intervals(preds_by_class, gts_by_class, LABELS, iou_thr=0.5)
    return float(mean_ap(ap_by_class))


def compute_fold_ensemble_ap(fold: int, w_cnn: float) -> float:
    folds = pd.read_csv("artifacts/splits/defog_folds.csv")
    val_paths = folds[folds["fold"] == fold]["path"].tolist()
    if not val_paths:
        return 0.0
    window_s = 5
    overlap = 0.5
    ds = WindowDataset(val_paths, window_s=window_s, overlap=overlap, sample_rate_hz=FPS, label_cols=LABELS)

    cnn = build_model("cnn_bilstm", num_classes=len(LABELS))
    tcn = build_model("tcn", num_classes=len(LABELS))
    for name, m in [("cnn_bilstm", cnn), ("tcn", tcn)]:
        ckpt = f"artifacts/checkpoints/{name}_fold{fold}_best.pt"
        if os.path.exists(ckpt):
            state = torch.load(ckpt, map_location="cpu")
            try:
                m.load_state_dict(state, strict=False)
            except Exception:
                if isinstance(state, dict) and "state_dict" in state:
                    m.load_state_dict(state["state_dict"], strict=False)
        m.eval()

    thresholds = load_fold_thresholds(fold, domain="defog")
    calibrators = load_fold_calibrators(fold)
    preds_by_class = {c: [] for c in LABELS}
    gts_by_class = {c: [] for c in LABELS}
    step = int(window_s * FPS)
    hop = int(step * (1.0 - overlap))
    with torch.no_grad():
        for k in range(len(ds)):
            x, y = ds[k]
            fi, wi = ds.index[k]
            x = x.unsqueeze(0)
            p1 = torch.sigmoid(cnn(x)).squeeze(0).cpu().numpy()
            p2 = torch.sigmoid(tcn(x)).squeeze(0).cpu().numpy()
            pw = (w_cnn * p1 + (1 - w_cnn) * p2)  # (T, K)
            scores = pw.max(axis=0)
            if calibrators:
                for ci, c in enumerate(LABELS):
                    if c in calibrators:
                        scores[ci] = float(calibrators[c].transform([scores[ci]])[0])
            start_s = (wi * hop) / FPS
            end_s = start_s + (step / FPS)
            for ci, c in enumerate(LABELS):
                if scores[ci] >= thresholds.get(c, 0.5):
                    preds_by_class[c].append((start_s, end_s, float(scores[ci])))
                if y.numel() and y[ci].item() >= 0.5:
                    gts_by_class[c].append((start_s, end_s))

    ap_by_class = score_intervals(preds_by_class, gts_by_class, LABELS, iou_thr=0.5)
    return float(mean_ap(ap_by_class))


def main():
    logger = get_logger("cv_metrics")
    per_fold_cnn = {}
    per_fold_tcn = {}

    for f in range(5):
        per_fold_cnn[f] = compute_fold_ap("cnn_bilstm", f)
        per_fold_tcn[f] = compute_fold_ap("tcn", f)
        logger.info({"fold": f, "cnn_ap": per_fold_cnn[f], "tcn_ap": per_fold_tcn[f]})

    cnn_macro = sum(per_fold_cnn.values()) / max(1, len(per_fold_cnn))
    tcn_macro = sum(per_fold_tcn.values()) / max(1, len(per_fold_tcn))

    # Ensemble weight
    wfile = "artifacts/ensemble/weights.json"
    w_cnn = 0.5
    if os.path.exists(wfile):
        with open(wfile, "r", encoding="utf-8") as f:
            w = json.load(f)
        if "cnn_bilstm" in w and "tcn" in w:
            w_cnn = float(w["cnn_bilstm"])

    per_fold_ens = {}
    for f in range(5):
        per_fold_ens[f] = compute_fold_ensemble_ap(f, w_cnn)
        logger.info({"fold": f, "ens_ap": per_fold_ens[f], "w_cnn": w_cnn})
    ens_macro = sum(per_fold_ens.values()) / max(1, len(per_fold_ens))

    payload = {
        "cv": {
            "macro_ap": ens_macro,
            "cnn": cnn_macro,
            "tcn": tcn_macro,
            "ensemble_w_cnn": w_cnn,
        }
    }
    os.makedirs("artifacts/metrics", exist_ok=True)
    with open("artifacts/metrics/metrics.json", "w", encoding="utf-8") as f:
        json.dump(payload, f)
    # Copy to website for display
    try:
        os.makedirs("website/netlify", exist_ok=True)
        with open("website/netlify/metrics.json", "w", encoding="utf-8") as f:
            json.dump(payload, f)
    except Exception:
        pass
    logger.info(payload)


if __name__ == "__main__":
    main()


