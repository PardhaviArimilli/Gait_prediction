import os
import joblib
import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader
from src.data.dataset import WindowDataset
from src.models.registry import build_model
from sklearn.isotonic import IsotonicRegression
from src.common.logging_utils import get_logger


LABELS = ["StartHesitation", "Turn", "Walking"]
FPS = 100.0


def fit_fold(fold: int, agg: str = "max"):
    logger = get_logger("calib_all")
    folds = pd.read_csv("artifacts/splits/defog_folds.csv")
    val_paths = folds[folds["fold"] == fold]["path"].tolist()
    if not val_paths:
        return
    ds = WindowDataset(val_paths, window_s=5.0, overlap=0.5, sample_rate_hz=FPS, label_cols=LABELS)
    dl = DataLoader(ds, batch_size=16, shuffle=False)

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

    scores_list, labels_list = [], []
    with torch.no_grad():
        for x, y in dl:
            logits = model(x)              # (B, T, K)
            probs = torch.sigmoid(logits)  # (B, T, K)
            if agg == "max":
                s = probs.max(dim=1).values
            else:
                s = probs.mean(dim=1)
            scores_list.append(s.cpu().numpy())
            labels_list.append(y.cpu().numpy())
    scores = np.concatenate(scores_list, axis=0)  # (N, K)
    labels = np.concatenate(labels_list, axis=0)  # (N, K)

    os.makedirs("artifacts/postprocess", exist_ok=True)
    for ci, c in enumerate(LABELS):
        try:
            ir = IsotonicRegression(out_of_bounds='clip')
            ir.fit(scores[:, ci], labels[:, ci])
            joblib.dump(ir, f"artifacts/postprocess/iso_fold{fold}_{c}.joblib")
        except Exception:
            logger.warning({"fold": fold, "class": c, "msg": "calibration skipped"})
    logger.info({"fold": fold, "msg": "saved isotonic calibrators"})


def main():
    for f in range(5):
        fit_fold(f, agg="max")


if __name__ == "__main__":
    main()


