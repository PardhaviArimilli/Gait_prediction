import numpy as np
import pandas as pd
from typing import Dict, List


def _lower_map(columns: List[str]) -> Dict[str, str]:
    """Map lowercased stripped column names to original names."""
    return {c.strip().lower(): c for c in columns}


def normalize_to_three_channels(df: pd.DataFrame) -> np.ndarray:
    """
    Normalize different CSV schemas to a 3-channel numeric array (T, 3).

    Supported schemas:
    - Accelerometer: AccV, AccML, AccAP (case-insensitive)
    - Gait params: {Cycle, stance_right, swing_right, stance_left, swing_left, step_length, step_width}
      Optional extra columns may include 'Dataset' (first) and "Normal/Parkinson's Disease" (last), which are ignored.

    Returns zeros if no supported schema is detected.
    """
    if df is None or df.empty:
        return np.zeros((0, 3), dtype=np.float32)

    colmap = _lower_map(list(df.columns))

    # 1) Try accelerometer schema
    acc_cols = ["accv", "accml", "accap"]
    if all(c in colmap for c in acc_cols):
        cols = [colmap[c] for c in acc_cols]
        arr = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
        return _clean_numeric(arr)

    # 2) Try gait-parameter schema
    gait_required = {"cycle", "stance_right", "swing_right", "stance_left", "swing_left", "step_length", "step_width"}
    if gait_required.issubset(set(colmap.keys())):
        # Drop label-like columns if present
        to_ignore = []
        for ign in ["dataset", "normal/parkinson's disease", "normal/parkinson’s disease"]:
            if ign in colmap:
                to_ignore.append(colmap[ign])
        if to_ignore:
            df = df.drop(columns=[c for c in to_ignore if c in df.columns])
            colmap = _lower_map(list(df.columns))

        step_length = df[colmap["step_length"]].apply(pd.to_numeric, errors="coerce")
        step_width = df[colmap["step_width"]].apply(pd.to_numeric, errors="coerce")
        stance_right = df[colmap["stance_right"]].apply(pd.to_numeric, errors="coerce")
        stance_left = df[colmap["stance_left"]].apply(pd.to_numeric, errors="coerce")
        # Channel 3: stance balance (right - left)
        stance_balance = stance_right - stance_left

        arr = np.stack([
            step_length.to_numpy(dtype=np.float32),
            step_width.to_numpy(dtype=np.float32),
            stance_balance.to_numpy(dtype=np.float32),
        ], axis=1)
        return _clean_numeric(arr)

    # 3) Fallback: pick first 3 non-time-like numeric columns
    exclude = {"time", "timestamp"}
    numeric_cols: List[str] = []
    for c in df.columns:
        if c.strip().lower() in exclude:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().any():
            numeric_cols.append(c)
        if len(numeric_cols) == 3:
            break
    if len(numeric_cols) < 3:
        # pad with zeros to 3 columns
        while len(numeric_cols) < 3:
            pad_name = f"pad_{len(numeric_cols)}"
            df[pad_name] = 0.0
            numeric_cols.append(pad_name)
    arr = df[numeric_cols[:3]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    return _clean_numeric(arr)


def _clean_numeric(arr: np.ndarray) -> np.ndarray:
    """Fill NaNs/Infs and ensure float32 array."""
    if arr.size == 0:
        return arr.astype(np.float32)
    arr = arr.astype(np.float32)
    mask = ~np.isfinite(arr)
    if mask.any():
        arr[mask] = 0.0
    return arr


