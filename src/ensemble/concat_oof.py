import os
import pandas as pd
from src.common.logging_utils import get_logger


def main():
    logger = get_logger("concat_oof")
    root = "artifacts/oof"
    files = [os.path.join(root, f) for f in os.listdir(root) if f.endswith('.csv')]
    if not files:
        logger.info("no oof files found")
        return
    parts = [pd.read_csv(f) for f in files]
    out = pd.concat(parts, ignore_index=True)
    out.to_csv(os.path.join(root, "oof_all.csv"), index=False)
    logger.info({"oof_all_shape": out.shape, "files": files})


if __name__ == "__main__":
    main()
