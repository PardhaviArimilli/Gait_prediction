import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.data.dataset import WindowDataset
from src.models.registry import build_model
from src.common.env import set_seed
from src.common.logging_utils import get_logger


def read_sample_paths():
    d_root = "train/defog"
    t_root = "train/tdcsfog"
    defog = [os.path.join(d_root, f) for f in os.listdir(d_root) if f.endswith('.csv')]
    tdcs = [os.path.join(t_root, f) for f in os.listdir(t_root) if f.endswith('.csv')]
    return defog, tdcs


def has_cols(path: str, cols):
    try:
        df = pd.read_csv(path, nrows=1)
        return all(c in df.columns for c in cols)
    except Exception:
        return False


def main(epochs: int = 1):
    logger = get_logger("mixed_train")
    set_seed(42)
    labels3 = ["StartHesitation", "Turn", "Walking"]
    defog_paths, tdcs_paths = read_sample_paths()
    defog_paths = [p for p in defog_paths if has_cols(p, labels3)]
    tdcs_paths = [p for p in tdcs_paths if has_cols(p, labels3)]
    if not defog_paths or not tdcs_paths:
        logger.info("need both defog and tdcs paths with 3-class labels")
        return

    defog_ds = WindowDataset(defog_paths, window_s=5, overlap=0.5, sample_rate_hz=100, label_cols=labels3)
    tdcs_ds = WindowDataset(tdcs_paths, window_s=5, overlap=0.5, sample_rate_hz=100, label_cols=labels3)
    defog_dl = DataLoader(defog_ds, batch_size=16, shuffle=True)
    tdcs_dl = DataLoader(tdcs_ds, batch_size=16, shuffle=True)

    model = build_model("cnn_bilstm", num_classes=3)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.BCEWithLogitsLoss()

    for ep in range(epochs):
        model.train()
        losses = []
        for (xb, yb), (xt, yt) in zip(defog_dl, tdcs_dl):
            opt.zero_grad()
            loss_b = crit(model(xb).mean(dim=1), yb)
            loss_t = crit(model(xt).mean(dim=1), yt)
            loss = loss_b + loss_t
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        logger.info({"epoch": ep, "mixed_train_loss": float(np.mean(losses))})


if __name__ == "__main__":
    main()
