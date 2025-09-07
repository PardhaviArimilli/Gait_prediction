import os
import json
import numpy as np
import pandas as pd
from src.eval.window_ap import window_ap
from src.common.logging_utils import get_logger


def main(fold: int = 0):
    logger = get_logger("ens_report")
    labels = ["StartHesitation", "Turn", "Walking"]
    oof_dir = "artifacts/oof"
    paths = {
        "cnn": os.path.join(oof_dir, f"cnn_bilstm_fold{fold}.csv"),
        "tcn": os.path.join(oof_dir, f"tcn_fold{fold}.csv"),
    }
    if not (os.path.exists(paths["cnn"]) and os.path.exists(paths["tcn"])):
        logger.info("missing per-model fold OOF files")
        return
    p1 = pd.read_csv(paths["cnn"]).values
    p2 = pd.read_csv(paths["tcn"]).values
    n = min(len(p1), len(p2))
    p1, p2 = p1[:n], p2[:n]

    # load weights
    wfile = "artifacts/ensemble/weights.json"
    if os.path.exists(wfile):
        with open(wfile, "r", encoding="utf-8") as f:
            weights = json.load(f)
        # Support both legacy per-file keys and new model keys
        if "cnn_bilstm" in weights:
            w_cnn = float(weights["cnn_bilstm"])
        elif len(weights) == 1:
            w_cnn = float(list(weights.values())[0])
        else:
            # try legacy specific key fallback
            w_cnn = float(weights.get("cnn_bilstm_fold0", 0.5))
    else:
        w_cnn = 0.5
    mix = w_cnn * p1 + (1 - w_cnn) * p2

    # No labels here; report mean probs as a proxy, and save mixed OOF for later scoring when labels are aligned
    os.makedirs("artifacts/oof", exist_ok=True)
    pd.DataFrame(mix, columns=labels).to_csv(os.path.join(oof_dir, f"ensemble_fold{fold}.csv"), index=False)
    score_proxy = float(mix.mean())
    logger.info({"fold": fold, "ensemble_mean_prob": score_proxy, "w_cnn": w_cnn})


if __name__ == "__main__":
    main()
