import os
import pandas as pd
from src.common.logging_utils import get_logger
from src.train.run_fold_train import main as train_fold
from src.train.run_fold_oof import main as oof_fold
from src.train.run_fold_oof_tcn import main as oof_fold_tcn
from src.train.run_fold_train_tcn import main as train_fold_tcn


def main(folds_to_run=(0, 1)):
    logger = get_logger("all_folds")
    if not os.path.exists("artifacts/splits/defog_folds.csv"):
        logger.info("missing defog splits; please run src/data/make_splits.py")
        return
    folds = pd.read_csv("artifacts/splits/defog_folds.csv")["fold"].unique()
    logger.info({"available_folds": list(map(int, folds))})
    for f in folds_to_run:
        logger.info({"fold": f, "stage": "train_cnn"})
        train_fold(f, epochs=20)
        logger.info({"fold": f, "stage": "oof_cnn"})
        oof_fold(f)

        logger.info({"fold": f, "stage": "train_tcn"})
        train_fold_tcn(f, epochs=20)
        logger.info({"fold": f, "stage": "oof_tcn"})
        oof_fold_tcn(f)


if __name__ == "__main__":
    main()
