import os
import joblib
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from src.data.dataset import WindowDataset
from src.models.registry import build_model
from src.eval.window_ap import window_ap
from src.common.logging_utils import get_logger


def main(fold: int = 0):
    logger = get_logger("report")
    labels = ["StartHesitation", "Turn", "Walking"]
    folds = pd.read_csv("artifacts/splits/defog_folds.csv")
    val_paths = folds[folds["fold"] == fold]["path"].tolist()[:50]
    ds = WindowDataset(val_paths, window_s=5, overlap=0.5, sample_rate_hz=100, label_cols=labels)
    dl = DataLoader(ds, batch_size=32, shuffle=False)

    model = build_model("cnn_bilstm", num_classes=3)
    model.load_state_dict(torch.load(f"artifacts/checkpoints/cnn_bilstm_fold{fold}_best.pt", map_location="cpu"))
    model.eval()

    # load calibration and thresholds
    th_path = f"artifacts/postprocess/thresholds_fold{fold}.json"
    if os.path.exists(th_path):
        with open(th_path, "r", encoding="utf-8") as f:
            thresholds = json.load(f)
    else:
        thresholds = {c: 0.5 for c in labels}
    calibrators = {}
    for c in labels:
        p = f"artifacts/postprocess/iso_fold{fold}_{c}.joblib"
        if os.path.exists(p):
            calibrators[c] = joblib.load(p)

    probs_all, labs_all = [], []
    with torch.no_grad():
        for x, y in dl:
            logits = model(x)
            probs = torch.sigmoid(logits.mean(dim=1)).cpu().numpy()
            # apply isotonic per class if available
            for ci, c in enumerate(labels):
                if c in calibrators:
                    probs[:, ci] = calibrators[c].transform(probs[:, ci])
            probs_all.append(probs)
            labs_all.append(y.cpu().numpy())
    probs = np.concatenate(probs_all, axis=0)
    labs = np.concatenate(labs_all, axis=0)

    # compute window AP
    wap = window_ap(probs, labs)

    os.makedirs("artifacts/reports", exist_ok=True)
    pd.DataFrame({"metric": ["window_ap"], "value": [wap]}).to_csv(
        f"artifacts/reports/fold{fold}_report.csv", index=False
    )
    logger.info({"fold": fold, "window_ap": wap, "thresholds": thresholds})


if __name__ == "__main__":
    main()
