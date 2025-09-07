import numpy as np
import pandas as pd
from typing import Tuple, List
from .schema import normalize_to_three_channels


def make_windows(df: pd.DataFrame, window_s: float, overlap: float, sample_rate_hz: float,
                 label_cols: List[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    label_cols = label_cols or []
    step = int(window_s * sample_rate_hz)
    hop = int(step * (1.0 - overlap))

    # Prefer original accelerometer columns when present (training data path).
    lower_cols = {c.lower() for c in df.columns}
    if {"accv", "accml", "accap"}.issubset(lower_cols):
        cols = []
        for key in ["accv", "accml", "accap"]:
            # find original casing
            orig = next(c for c in df.columns if c.lower() == key)
            cols.append(orig)
        data = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    else:
        # Inference or alternative schema path: normalize to 3 channels
        data = normalize_to_three_channels(df)

    labels_arr = df[label_cols].values.astype(np.float32) if label_cols else None

    X, Y = [], []
    for start in range(0, len(df) - step + 1, max(1, hop)):
        end = start + step
        X.append(data[start:end, :])
        if labels_arr is not None:
            # window is positive if any frame is positive in the window (per class)
            win_lab = labels_arr[start:end]
            Y.append((win_lab.max(axis=0) > 0.0).astype(np.float32))
    X = np.stack(X) if X else np.empty((0, step, 3), dtype=np.float32)
    if labels_arr is None:
        return X, np.empty((0, 0), dtype=np.float32)
    Y = np.stack(Y) if Y else np.empty((0, labels_arr.shape[1]), dtype=np.float32)
    return X, Y
