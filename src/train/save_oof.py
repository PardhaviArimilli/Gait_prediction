import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.data.dataset import WindowDataset
from src.models.registry import build_model
from src.common.env import set_seed
from src.common.logging_utils import get_logger


def main():
    logger = get_logger("oof")
    set_seed(42)
    root = "train/defog"
    files = [os.path.join(root, f) for f in os.listdir(root) if f.endswith('.csv')][:2]
    if not files:
        logger.info("no files for OOF")
        return
    ds = WindowDataset(files, window_s=5, overlap=0.5, sample_rate_hz=100,
                       label_cols=["StartHesitation", "Turn", "Walking"])
    dl = DataLoader(ds, batch_size=8, shuffle=False)

    model = build_model("cnn_bilstm", num_classes=3)
    model.eval()

    all_probs = []
    with torch.no_grad():
        for x, _ in dl:
            logits = model(x)
            probs = torch.sigmoid(logits.mean(dim=1)).cpu().numpy()
            all_probs.append(probs)
    probs = np.concatenate(all_probs, axis=0) if all_probs else np.zeros((0, 3))
    os.makedirs("artifacts/oof", exist_ok=True)
    out = pd.DataFrame(probs, columns=["StartHesitation", "Turn", "Walking"]) 
    out.to_csv("artifacts/oof/cnn_bilstm_oof.csv", index=False)
    logger.info(f"saved OOF probs: {out.shape}")


if __name__ == "__main__":
    main()
