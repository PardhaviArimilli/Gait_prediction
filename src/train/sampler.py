import numpy as np
from typing import Tuple


def select_indices_by_positive_ratio(Y: np.ndarray, positive_ratio: float = 0.4) -> np.ndarray:
    """
    Y: (N, K) binary labels per window (any class)
    Returns indices selecting approximately positive_ratio positives.
    """
    if Y.size == 0:
        return np.arange(0)
    pos_mask = (Y.max(axis=1) > 0)
    pos_idx = np.where(pos_mask)[0]
    neg_idx = np.where(~pos_mask)[0]
    n = len(Y)
    n_pos_target = int(n * positive_ratio)
    n_pos = min(n_pos_target, len(pos_idx))
    n_neg = max(0, n - n_pos)
    sel_pos = np.random.choice(pos_idx, size=n_pos, replace=len(pos_idx) < n_pos)
    sel_neg = np.random.choice(neg_idx, size=n_neg, replace=len(neg_idx) < n_neg)
    sel = np.concatenate([sel_pos, sel_neg])
    np.random.shuffle(sel)
    return sel
