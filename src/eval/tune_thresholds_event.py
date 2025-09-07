import os
import json
import numpy as np
import pandas as pd
import torch

from src.data.dataset import WindowDataset
from src.models.registry import build_model
from src.eval.scorer import score_intervals, mean_ap
from src.common.logging_utils import get_logger


LABELS = ["StartHesitation", "Turn", "Walking"]
FPS = 100.0


def collect_window_spans_and_scores(model, ds, window_s: float, overlap: float):
    step = int(window_s * FPS)
    hop = int(step * (1.0 - overlap))
    spans = []  # list of (start_s, end_s, scores[K], labels[K]) per window
    with torch.no_grad():
        for k in range(len(ds)):
            x, y = ds[k]
            fi, wi = ds.index[k]
            x = x.unsqueeze(0)
            logits = model(x)
            pw = torch.sigmoid(logits).squeeze(0).cpu().numpy()  # (T, K)
            scores = pw.max(axis=0)  # (K,)
            yv = y.numpy() if y.numel() else np.zeros(len(LABELS), dtype=np.float32)
            start_s = (wi * hop) / FPS
            end_s = start_s + (step / FPS)
            spans.append((start_s, end_s, scores.astype(float), yv.astype(float)))
    return spans


def build_gts_from_spans(spans):
    gts_by_class = {c: [] for c in LABELS}
    for (s, e, _, yv) in spans:
        for ci, c in enumerate(LABELS):
            if yv[ci] >= 0.5:
                gts_by_class[c].append((s, e))
    return gts_by_class


def predict_from_spans(spans, thresholds: dict):
    preds_by_class = {c: [] for c in LABELS}
    for (s, e, sc, _) in spans:
        for ci, c in enumerate(LABELS):
            if sc[ci] >= thresholds.get(c, 0.5):
                preds_by_class[c].append((s, e, float(sc[ci])))
    # Optionally, merge adjacent window intervals of same class
    merged = {}
    for c in LABELS:
        items = sorted(preds_by_class[c])
        out = []
        for (s, e, sc) in items:
            if not out:
                out.append([s, e, sc])
            else:
                ps, pe, psc = out[-1]
                if s <= pe:  # adjoining/overlap
                    out[-1][1] = max(pe, e)
                    out[-1][2] = max(psc, sc)
                else:
                    out.append([s, e, sc])
        merged[c] = [(s, e, sc) for s, e, sc in out]
    return merged


def tune_thresholds_for_fold(fold: int) -> dict:
    logger = get_logger("tune_thr")
    folds = pd.read_csv("artifacts/splits/defog_folds.csv")
    val_paths = folds[folds["fold"] == fold]["path"].tolist()
    if not val_paths:
        return {c: 0.5 for c in LABELS}
    window_s = 5.0
    overlap = 0.5
    ds = WindowDataset(val_paths, window_s=window_s, overlap=overlap, sample_rate_hz=FPS, label_cols=LABELS)

    model = build_model("cnn_bilstm", num_classes=len(LABELS))
    ckpt = f"artifacts/checkpoints/cnn_bilstm_fold{fold}_best.pt"
    if os.path.exists(ckpt):
        state = torch.load(ckpt, map_location="cpu")
        try:
            model.load_state_dict(state, strict=False)
        except Exception:
            if isinstance(state, dict) and "state_dict" in state:
                model.load_state_dict(state["state_dict"], strict=False)
    model.eval()

    spans = collect_window_spans_and_scores(model, ds, window_s, overlap)
    gts = build_gts_from_spans(spans)

    # Coordinate descent on thresholds per class
    thr = {c: 0.5 for c in LABELS}
    grid = np.linspace(0.2, 0.8, 13)  # step 0.05
    improved = True
    it = 0
    while improved and it < 3:
        improved = False
        it += 1
        for c in LABELS:
            best_ap = -1.0
            best_t = thr[c]
            for t in grid:
                temp = dict(thr)
                temp[c] = float(t)
                preds = predict_from_spans(spans, temp)
                ap_by = score_intervals(preds, gts, LABELS, iou_thr=0.5)
                mapv = mean_ap(ap_by)
                if mapv > best_ap:
                    best_ap = mapv
                    best_t = float(t)
            if best_t != thr[c]:
                thr[c] = best_t
                improved = True
        logger.info({"fold": fold, "iter": it, "thr": thr})
    return thr


def main():
    os.makedirs("artifacts/postprocess", exist_ok=True)
    all_thr = {}
    for f in range(5):
        thr = tune_thresholds_for_fold(f)
        all_thr[f] = thr
        with open(f"artifacts/postprocess/defog_fold{f}_thresholds.json", "w", encoding="utf-8") as w:
            json.dump(thr, w)
    # Save a summary
    with open("artifacts/postprocess/thresholds_summary.json", "w", encoding="utf-8") as w:
        json.dump(all_thr, w)


if __name__ == "__main__":
    main()


