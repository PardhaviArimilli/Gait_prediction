import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from src.common.logging_utils import get_logger
from src.common.env import set_seed
from src.models.registry import build_model
from src.data.dataset import WindowDataset


def load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main(args):
    logger = get_logger("train")
    data_cfg = load_yaml(args.data_cfg)
    train_cfg = load_yaml(args.train_cfg)
    model_cfg = load_yaml(args.model_cfg)

    set_seed(train_cfg.get("seed", 42))

    logger.info({"data_cfg": data_cfg})
    logger.info({"train_cfg": train_cfg})
    logger.info({"model_cfg": model_cfg})

    model_name = model_cfg.get("name", "cnn_bilstm")
    model_kwargs = {k: v for k, v in model_cfg.items() if k != "name"}
    model = build_model(model_name, **model_kwargs)

    # create tiny dataset from CSVs that contain required labels (defog schema)
    import os
    import pandas as pd

    required_labels = ["StartHesitation", "Turn", "Walking"]

    def list_csvs(root):
        if not os.path.isdir(root):
            return []
        return [os.path.join(root, f) for f in os.listdir(root) if f.endswith('.csv')]

    def has_required_cols(path):
        try:
            df = pd.read_csv(path, nrows=1)
            return all(c in df.columns for c in required_labels)
        except Exception:
            return False

    candidates = list_csvs(data_cfg["paths"]["raw_train_defog"]) + list_csvs(data_cfg["paths"]["raw_train_tdcs"])
    files = [p for p in candidates if has_required_cols(p)][:2]

    if not files:
        logger.info("no labeled CSV files with required columns found; skipping dataset loading")
        return
    ds = WindowDataset(files, window_s=data_cfg["window_s"], overlap=data_cfg["overlap"],
                       sample_rate_hz=data_cfg["sample_rate_hz"],
                       label_cols=required_labels)
    dl = DataLoader(ds, batch_size=4, shuffle=False)
    batch = next(iter(dl))
    x, y = batch
    logger.info(f"batch x shape: {tuple(x.shape)}, y shape: {tuple(y.shape)}")

    # forward pass
    logits = model(x)  # (B, T, K)
    logger.info(f"logits shape: {tuple(logits.shape)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_cfg", default="configs/data.yaml")
    parser.add_argument("--train_cfg", default="configs/training_baseline.yaml")
    parser.add_argument("--model_cfg", default="configs/model_cnn_bilstm.yaml")
    main(parser.parse_args())
