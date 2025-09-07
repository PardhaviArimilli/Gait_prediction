import numpy as np
from .scorer import average_precision


def window_ap(probs: np.ndarray, labels: np.ndarray) -> float:
    """
    probs: (N, K), labels: (N, K) binary
    Computes macro AP over classes via standard precision-recall.
    """
    aps = []
    for k in range(probs.shape[1]):
        p = probs[:, k]
        y = labels[:, k].astype(bool)
        if y.sum() == 0:
            aps.append(0.0)
            continue
        order = np.argsort(-p)
        y_sorted = y[order]
        tp = np.cumsum(y_sorted)
        fp = np.cumsum(~y_sorted)
        rec = tp / max(y.sum(), 1)
        prec = tp / np.maximum(tp + fp, 1e-9)
        aps.append(average_precision(rec, prec))
    return float(np.mean(aps)) if aps else 0.0


def write_cv_metrics_json(out_path: str, per_fold_scores: dict):
    import json
    import os
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    macro = sum(per_fold_scores.values()) / max(1, len(per_fold_scores))
    payload = {"cv": {"macro_ap": macro, **per_fold_scores}}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
