import os
import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.isotonic import IsotonicRegression
from src.data.dataset import WindowDataset
from src.models.registry import build_model
from src.common.logging_utils import get_logger


def main():
    logger = get_logger("calib")
    folds = pd.read_csv("artifacts/splits/defog_folds.csv")
    val_paths = folds[folds["fold"] == 0]["path"].tolist()[:20]
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

    os.makedirs("artifacts/postprocess", exist_ok=True)
    for ci, c in enumerate(labels):
        ir = IsotonicRegression(out_of_bounds='clip')
        ir.fit(probs[:, ci], labs[:, ci])
        joblib.dump(ir, f"artifacts/postprocess/iso_fold0_{c}.joblib")
    logger.info("saved isotonic calibrators for fold0")


if __name__ == "__main__":
    main()
