import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from src.data.dataset import WindowDataset
from src.models.registry import build_model
from src.eval.window_ap import window_ap
from src.common.logging_utils import get_logger


def main():
    logger = get_logger("tune")
    folds = pd.read_csv("artifacts/splits/defog_folds.csv")
    # tune per fold across all validation files
    best_by_fold = {}
    labels = ["StartHesitation", "Turn", "Walking"]
    for fold in sorted(folds["fold"].unique()):
        val_paths = folds[folds["fold"] == fold]["path"].tolist()
        ds = WindowDataset(val_paths, window_s=5, overlap=0.5, sample_rate_hz=100, label_cols=labels)
        dl = DataLoader(ds, batch_size=32, shuffle=False)

        model = build_model("cnn_bilstm", num_classes=3)
        ckpt = f"artifacts/checkpoints/cnn_bilstm_fold{fold}_best.pt"
        if not os.path.exists(ckpt):
            continue
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        model.eval()

        probs_all, labs_all = [], []
        with torch.no_grad():
            for x, y in dl:
                logits = model(x)
                probs = torch.sigmoid(logits.mean(dim=1)).cpu().numpy()
                probs_all.append(probs)
                labs_all.append(y.cpu().numpy())
        probs = np.concatenate(probs_all, axis=0)
        labs = np.concatenate(labs_all, axis=0)

        best = {}
        grid = np.linspace(0.2, 0.9, 36)
        for ci, c in enumerate(labels):
            best_ap, best_th = -1.0, 0.5
            for th in grid:
                binarized = probs.copy()
                binarized[:, ci] = (probs[:, ci] >= th).astype(float)
                ap = window_ap(binarized, labs)
                if ap > best_ap:
                    best_ap, best_th = ap, th
            best[c] = float(best_th)
        os.makedirs("artifacts/postprocess", exist_ok=True)
        out = f"artifacts/postprocess/thresholds_fold{fold}.json"
        pd.Series(best).to_json(out)
        best_by_fold[fold] = best
        logger.info({"fold": int(fold), "best_thresholds": best})

    labels = ["StartHesitation", "Turn", "Walking"]
    ds = WindowDataset(val_paths, window_s=5, overlap=0.5, sample_rate_hz=100, label_cols=labels)
    dl = DataLoader(ds, batch_size=32, shuffle=False)

    model = build_model("cnn_bilstm", num_classes=3)
    model.load_state_dict(torch.load("artifacts/checkpoints/cnn_bilstm_fold0_best.pt", map_location="cpu"))
    model.eval()

    probs_all, labs_all = [], []
    with torch.no_grad():
        for x, y in dl:
            logits = model(x)
            probs = torch.sigmoid(logits.mean(dim=1)).cpu().numpy()
            probs_all.append(probs)
            labs_all.append(y.cpu().numpy())
    probs = np.concatenate(probs_all, axis=0)
    labs = np.concatenate(labs_all, axis=0)

    # simple scalar threshold per class tuned to maximize window AP
    best = {}
    grid = np.linspace(0.3, 0.8, 11)
    for ci, c in enumerate(labels):
        best_ap, best_th = -1.0, 0.5
        for th in grid:
            binarized = probs.copy()
            binarized[:, ci] = (probs[:, ci] >= th).astype(float)
            ap = window_ap(binarized, labs)
            if ap > best_ap:
                best_ap, best_th = ap, th
        best[c] = float(best_th)
    os.makedirs("artifacts/postprocess", exist_ok=True)
    pd.Series(best).to_json("artifacts/postprocess/thresholds_fold0.json")
    logger.info({"best_thresholds": best})


if __name__ == "__main__":
    main()
