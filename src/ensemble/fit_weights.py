import os
import json
import glob
import numpy as np
import pandas as pd
from src.common.logging_utils import get_logger
from src.eval.window_ap import window_ap
from src.data.dataset import WindowDataset
from torch.utils.data import DataLoader
from src.models.registry import build_model
import torch


def ap_for_weights(w: float, a: np.ndarray, b: np.ndarray, labels: np.ndarray) -> float:
    mix = w * a + (1 - w) * b
    return float(window_ap(mix, labels))


def main():
    logger = get_logger("ensemble")
    oof_dir = "artifacts/oof"
    # Gather per-fold pairs for cnn_bilstm and tcn
    cnn_paths = {os.path.splitext(os.path.basename(p))[0].split('_fold')[-1]: p
                 for p in glob.glob(os.path.join(oof_dir, "cnn_bilstm_fold*.csv"))}
    tcn_paths = {os.path.splitext(os.path.basename(p))[0].split('_fold')[-1]: p
                 for p in glob.glob(os.path.join(oof_dir, "tcn_fold*.csv"))}
    common_folds = sorted(set(cnn_paths.keys()) & set(tcn_paths.keys()))
    if not common_folds:
        # Fallback: pick any two files if available
        paths = [p for p in os.listdir(oof_dir) if p.endswith('.csv') and not p.startswith('oof_all')]
        if len(paths) < 2:
            weights = {paths[0].split('.')[0]: 1.0} if paths else {"cnn_bilstm": 1.0}
        else:
            a = pd.read_csv(os.path.join(oof_dir, paths[0])).values
            b = pd.read_csv(os.path.join(oof_dir, paths[1])).values
            n = min(len(a), len(b))
            a, b = a[:n], b[:n]
            best_w, best_score = 0.5, -1.0
            for w in np.linspace(0.0, 1.0, 21):
                mix = w * a + (1 - w) * b
                score = mean_prob_ap(mix)
                if score > best_score:
                    best_score, best_w = score, w
            weights = {paths[0].split('.')[0]: float(best_w), paths[1].split('.')[0]: float(1 - best_w)}
    else:
        # Use true labels from validation windows per fold to directly optimize AP
        labels_cols = ["StartHesitation", "Turn", "Walking"]
        A_list, B_list, Y_list = [], [], []
        folds_df = pd.read_csv("artifacts/splits/defog_folds.csv")
        for f in common_folds:
            a = pd.read_csv(cnn_paths[f]).values
            b = pd.read_csv(tcn_paths[f]).values
            # Recompute labels for the exact same windowing to align shapes
            val_paths = folds_df[folds_df["fold"] == int(f)]["path"].tolist()
            ds = WindowDataset(val_paths, window_s=5, overlap=0.5, sample_rate_hz=100, label_cols=labels_cols)
            dl = DataLoader(ds, batch_size=64, shuffle=False)
            labs_all = []
            for _, y in dl:
                labs_all.append(y.numpy())
            labs = np.concatenate(labs_all, axis=0) if labs_all else np.zeros_like(a)
            n = min(len(a), len(b), len(labs))
            A_list.append(a[:n])
            B_list.append(b[:n])
            Y_list.append(labs[:n])
        A = np.concatenate(A_list, axis=0)
        B = np.concatenate(B_list, axis=0)
        Y = np.concatenate(Y_list, axis=0)
        best_w, best_score = 0.5, -1.0
        for w in np.linspace(0.0, 1.0, 41):
            score = ap_for_weights(w, A, B, Y)
            if score > best_score:
                best_score, best_w = score, w
        weights = {"cnn_bilstm": float(best_w), "tcn": float(1 - best_w)}
    os.makedirs("artifacts/ensemble", exist_ok=True)
    with open("artifacts/ensemble/weights.json", "w", encoding="utf-8") as f:
        json.dump(weights, f)
    logger.info({"weights": weights})


if __name__ == "__main__":
    main()
