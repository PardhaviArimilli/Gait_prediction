import os
import pandas as pd
from sklearn.model_selection import GroupKFold
from src.common.logging_utils import get_logger


def main(n_splits: int = 5):
    logger = get_logger("splits_tdcs")
    root = "train/tdcsfog"
    files = [os.path.join(root, f) for f in os.listdir(root) if f.endswith('.csv')]
    if not files:
        logger.info("no files for tdcs splits")
        return
    meta = pd.read_csv("tdcsfog_metadata.csv")  # Id, Subject, Visit, Test, Medication
    df = pd.DataFrame({"path": files})
    df["Id"] = df["path"].apply(lambda p: os.path.splitext(os.path.basename(p))[0])
    df = df.merge(meta[["Id", "Subject"]], on="Id", how="left")
    df["Subject"].fillna(df["Id"], inplace=True)

    gkf = GroupKFold(n_splits=n_splits)
    X = df.index.values
    y = df.index.values
    groups = df["Subject"].values
    splits = []
    for k, (tr, va) in enumerate(gkf.split(X, y, groups)):
        fold_df = pd.DataFrame({
            "path": df.loc[va, "path"].values,
            "fold": k,
            "subject": df.loc[va, "Subject"].values,
        })
        splits.append(fold_df)
    out = pd.concat(splits, ignore_index=True)
    os.makedirs("artifacts/splits", exist_ok=True)
    out.to_csv("artifacts/splits/tdcs_folds.csv", index=False)
    logger.info(f"saved tdcs splits: {out.shape}")


if __name__ == "__main__":
    main()
