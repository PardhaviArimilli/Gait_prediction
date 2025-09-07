import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.data.dataset import WindowDataset
from src.models.registry import build_model
from src.common.env import set_seed
from src.common.logging_utils import get_logger


def main(fold: int = 0):
    logger = get_logger("fold_oof")
    set_seed(42)
    folds = pd.read_csv("artifacts/splits/defog_folds.csv")
    val_paths = folds[folds["fold"] == fold]["path"].tolist()
    if not val_paths:
        logger.info("no val paths for fold")
        return
    ds = WindowDataset(val_paths, window_s=5, overlap=0.5, sample_rate_hz=100,
                       label_cols=["StartHesitation", "Turn", "Walking"])  # subset for speed
    dl = DataLoader(ds, batch_size=16, shuffle=False)
    model = build_model("cnn_bilstm", num_classes=3)
    # Load best checkpoint if available
    ckpt = f"artifacts/checkpoints/cnn_bilstm_fold{fold}_best.pt"
    if os.path.exists(ckpt):
        state = torch.load(ckpt, map_location="cpu")
        try:
            model.load_state_dict(state, strict=False)
        except Exception:
            if isinstance(state, dict) and "state_dict" in state:
                model.load_state_dict(state["state_dict"], strict=False)
    model.eval()
    probs_all = []
    with torch.no_grad():
        for x, _ in dl:
            logits = model(x)
            probs = torch.sigmoid(logits.mean(dim=1)).cpu().numpy()
            probs_all.append(probs)
    probs = np.concatenate(probs_all, axis=0) if probs_all else np.zeros((0, 3))
    os.makedirs("artifacts/oof", exist_ok=True)
    pd.DataFrame(probs, columns=["StartHesitation", "Turn", "Walking"]).to_csv(
        f"artifacts/oof/cnn_bilstm_fold{fold}.csv", index=False
    )
    logger.info(f"saved fold {fold} OOF probs: {probs.shape}")


if __name__ == "__main__":
    main()
