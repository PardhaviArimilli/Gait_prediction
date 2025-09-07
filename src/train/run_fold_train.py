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
from src.eval.window_ap import window_ap
from src.data.transforms import Compose, StandardizePerWindow, Jitter, Scale, Rotate3D
from src.losses.focal_bce import FocalBCEWithLogitsLoss
from src.train.sampler import select_indices_by_positive_ratio


def main(fold: int = 0, epochs: int = 3):
    logger = get_logger("fold_train")
    set_seed(42)
    folds = pd.read_csv("artifacts/splits/defog_folds.csv")
    val_paths = folds[folds["fold"] == fold]["path"].tolist()
    train_paths = folds[folds["fold"] != fold]["path"].tolist()
    if not val_paths or not train_paths:
        logger.info("insufficient paths for fold training")
        return
    labels = ["StartHesitation", "Turn", "Walking"]
    train_tf = Compose([StandardizePerWindow(), Jitter(0.01, 0.5), Scale(0.95, 1.05, 0.5), Rotate3D(5.0, 0.3)])
    val_tf = Compose([StandardizePerWindow()])
    train_ds = WindowDataset(train_paths, window_s=5, overlap=0.5, sample_rate_hz=100, label_cols=labels, transform=train_tf)
    val_ds = WindowDataset(val_paths, window_s=5, overlap=0.5, sample_rate_hz=100, label_cols=labels, transform=val_tf)
    # Positive-aware sampling: build indices per epoch
    def make_loader(ds, shuffle=True):
        X, Y = [], []
        for _, y in ds.cache:
            Y.append(y)
        import numpy as np
        if Y:
            Yc = np.concatenate(Y, axis=0)
            idx = select_indices_by_positive_ratio(Yc, positive_ratio=0.4)
        else:
            idx = None
        return DataLoader(ds, batch_size=32, shuffle=shuffle) if idx is None else DataLoader(ds, batch_size=32, sampler=idx.tolist())

    train_dl = make_loader(train_ds, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=32, shuffle=False)

    model = build_model("cnn_bilstm", num_classes=3)
    # Cosine schedule with warmup
    base_lr = 1e-3
    opt = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=1e-4)
    total_epochs = epochs
    warmup_epochs = max(1, epochs // 10)
    def get_lr(epoch):
        if epoch < warmup_epochs:
            return base_lr * (epoch + 1) / warmup_epochs
        t = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * base_lr * (1 + np.cos(np.pi * t))
    crit = FocalBCEWithLogitsLoss(gamma=1.5)

    best = -1.0
    patience, bad = 7, 0
    os.makedirs("artifacts/checkpoints", exist_ok=True)

    for ep in range(epochs):
        model.train()
        losses = []
        for x, y in train_dl:
            opt.zero_grad()
            logits = model(x)
            loss = crit(logits.mean(dim=1), y)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        # apply LR schedule
        for g in opt.param_groups:
            g['lr'] = get_lr(ep)
        logger.info({"fold": fold, "epoch": ep, "train_loss": float(np.mean(losses)), "lr": get_lr(ep)})

        # validation window AP
        model.eval()
        probs_all, labs_all = [], []
        with torch.no_grad():
            for x, y in val_dl:
                logits = model(x)
                probs = torch.sigmoid(logits.mean(dim=1)).cpu().numpy()
                probs_all.append(probs)
                labs_all.append(y.cpu().numpy())
        if probs_all:
            probs = np.concatenate(probs_all, axis=0)
            labs = np.concatenate(labs_all, axis=0)
            ap = window_ap(probs, labs)
            logger.info({"fold": fold, "epoch": ep, "val_window_ap": ap})
            if ap > best:
                best = ap
                torch.save(model.state_dict(), f"artifacts/checkpoints/cnn_bilstm_fold{fold}_best.pt")
                logger.info(f"saved best fold{fold} checkpoint ap={best:.4f}")
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    logger.info({"fold": fold, "early_stop": True, "best_ap": best})
                    break


if __name__ == "__main__":
    main()
