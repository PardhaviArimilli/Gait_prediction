import os
import pandas as pd
from src.common.logging_utils import get_logger

REQUIRED_COLUMNS = ["Time", "AccV", "AccML", "AccAP"]


def validate_csv(path: str) -> dict:
    df = pd.read_csv(path, nrows=20000)
    issues = []
    for c in REQUIRED_COLUMNS:
        if c not in df.columns:
            issues.append(f"missing_column:{c}")
    if "Time" in df.columns:
        if not (df["Time"].values[1:] >= df["Time"].values[:-1]).all():
            issues.append("non_monotonic_time")
    # crude sample rate estimate
    sr = None
    if "Time" in df.columns and len(df) > 1:
        diffs = df["Time"].diff().dropna()
        if not diffs.empty:
            sr = round(1.0 / diffs.mode().iloc[0], 2) if diffs.mode().size > 0 and diffs.mode().iloc[0] != 0 else None
    return {"path": path, "issues": issues, "sample_rate_hz": sr}


def main(root_defog: str, root_tdcs: str, log_dir: str = "logs") -> None:
    logger = get_logger("pipeline", log_dir)
    files = []
    for root in [root_defog, root_tdcs]:
        if os.path.isdir(root):
            for fname in os.listdir(root):
                if fname.endswith(".csv"):
                    files.append(os.path.join(root, fname))
    logger.info(f"validating {len(files)} files")
    for f in files[:200]:  # limit in first pass
        res = validate_csv(f)
        logger.info(res)


if __name__ == "__main__":
    main("train/defog", "train/tdcsfog")
