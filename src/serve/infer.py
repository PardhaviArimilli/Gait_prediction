import os
import argparse
import glob
import json
from typing import List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.models.registry import build_model
from src.data.dataset import WindowDataset
from src.common.env import set_seed
from src.common.logging_utils import get_logger


LABELS = ["StartHesitation", "Turn", "Walking"]


def infer_on_dir(input_dir: str, output_dir: str, checkpoint_path: str,
                 model_name: str = "cnn_bilstm", window_s: float = 5.0,
                 overlap: float = 0.5, sample_rate_hz: float = 100.0,
                 batch_size: int = 32) -> List[str]:
    logger = get_logger("infer")
    set_seed(42)

    files = sorted([p for p in glob.glob(os.path.join(input_dir, "*.csv"))])
    if not files:
        logger.info({"msg": "no input files", "input_dir": input_dir})
        return []

    # Build model and load checkpoint if provided
    model = build_model(model_name, num_classes=len(LABELS))
    if checkpoint_path and os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location="cpu")
        try:
            model.load_state_dict(state, strict=False)
        except Exception:
            # Fallback for state dicts saved as dict with 'state_dict'
            if isinstance(state, dict) and "state_dict" in state:
                model.load_state_dict(state["state_dict"], strict=False)
    model.eval()

    os.makedirs(output_dir, exist_ok=True)
    written: List[str] = []

    # WindowDataset with no labels for inference
    ds = WindowDataset(files, window_s=window_s, overlap=overlap,
                       sample_rate_hz=sample_rate_hz, label_cols=[])
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)

    # We need to map dataset indices back to file names
    file_windows: List[int] = []
    for X, _ in ds.cache:
        file_windows.append(len(X))
    file_offsets = np.cumsum([0] + file_windows[:-1])

    # Collect per-window probabilities in original order
    probs_all: List[np.ndarray] = []
    with torch.no_grad():
        for xb, _ in dl:
            logits = model(xb)
            probs = torch.sigmoid(logits).mean(dim=1).cpu().numpy()  # (B, K)
            probs_all.append(probs)
    probs_cat = np.concatenate(probs_all, axis=0) if probs_all else np.zeros((0, len(LABELS)))

    # Split back to files and write
    start = 0
    for idx_file, num_w in enumerate(file_windows):
        end = start + num_w
        file_probs = probs_cat[start:end]
        start = end
        base = os.path.splitext(os.path.basename(files[idx_file]))[0]
        out_path = os.path.join(output_dir, f"{base}_probs.csv")
        pd.DataFrame(file_probs, columns=LABELS).to_csv(out_path, index=False)
        written.append(out_path)
        logger.info({"written": out_path, "windows": int(num_w)})

    # Save manifest
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({"inputs": files, "outputs": written, "checkpoint": checkpoint_path,
                   "model": model_name, "labels": LABELS}, f)
    logger.info({"manifest": manifest_path, "count": len(written)})
    return written


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--checkpoint", default="")
    p.add_argument("--model", default="cnn_bilstm")
    p.add_argument("--window_s", type=float, default=5.0)
    p.add_argument("--overlap", type=float, default=0.5)
    p.add_argument("--sample_rate_hz", type=float, default=100.0)
    p.add_argument("--batch_size", type=int, default=32)
    return p.parse_args()


def main():
    args = parse_args()
    infer_on_dir(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        checkpoint_path=args.checkpoint,
        model_name=args.model,
        window_s=args.window_s,
        overlap=args.overlap,
        sample_rate_hz=args.sample_rate_hz,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()


