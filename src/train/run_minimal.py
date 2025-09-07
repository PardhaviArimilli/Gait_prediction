import os
import torch
from torch.utils.data import DataLoader
from src.common.logging_utils import get_logger
from src.common.env import set_seed
from src.data.dataset import WindowDataset
from src.models.registry import build_model
from .loop import train_one_epoch


def main():
    logger = get_logger("train_min")
    set_seed(42)
    root = "train/defog"
    files = [os.path.join(root, f) for f in os.listdir(root) if f.endswith('.csv')][:2]
    if not files:
        logger.info("no files for training")
        return
    ds = WindowDataset(files, window_s=5, overlap=0.5, sample_rate_hz=100,
                       label_cols=["StartHesitation", "Turn", "Walking"])
    dl = DataLoader(ds, batch_size=16, shuffle=True)
    model = build_model("cnn_bilstm", num_classes=3)

    best = float('inf')
    os.makedirs("artifacts/checkpoints", exist_ok=True)

    for epoch in range(3):  # few epochs
        metrics = train_one_epoch(model, dl)
        logger.info({"epoch": epoch, **metrics})
        if metrics["loss"] < best:
            best = metrics["loss"]
            torch.save(model.state_dict(), "artifacts/checkpoints/cnn_bilstm_min.pt")
            logger.info(f"saved checkpoint with loss={best:.4f}")


if __name__ == "__main__":
    main()
