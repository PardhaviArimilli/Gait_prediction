import os
import glob
import json
from typing import Dict, List

import numpy as np
import pandas as pd
import joblib

from src.serve.infer import infer_on_dir
from src.eval.postprocess import probs_to_intervals
import yaml
from src.common.logging_utils import get_logger


LABELS = ["StartHesitation", "Turn", "Walking"]


def load_average_thresholds(dir_path: str = "artifacts/postprocess") -> Dict[str, float]:
    by_fold: List[Dict[str, float]] = []
    for p in glob.glob(os.path.join(dir_path, "thresholds_fold*.json")):
        try:
            with open(p, "r", encoding="utf-8") as f:
                by_fold.append(json.load(f))
        except Exception:
            pass
    if not by_fold:
        # default conservative
        return {c: 0.3 for c in LABELS}
    avg = {}
    for c in LABELS:
        vals = [d.get(c, 0.3) for d in by_fold]
        avg[c] = float(np.mean(vals))
    return avg


def load_isotonic_calibrators(dir_path: str = "artifacts/postprocess") -> Dict[str, object]:
    cal: Dict[str, object] = {}
    for c in LABELS:
        # try fold-specific first, then any
        pattern = os.path.join(dir_path, f"iso_*_{c}.joblib")
        matches = sorted(glob.glob(pattern))
        if matches:
            try:
                cal[c] = joblib.load(matches[0])
            except Exception:
                pass
    return cal


def ensemble_and_postprocess(input_dir: str,
                             out_root: str,
                             ckpt_cnn: str,
                             ckpt_tcn: str,
                             weights_json: str = "artifacts/ensemble/weights.json",
                             window_s: float = 5.0,
                             overlap: float = 0.5) -> None:
    logger = get_logger("ens_infer")
    os.makedirs(out_root, exist_ok=True)

    # 1) Run per-model inference (supports fold averaging if multiple ckpts provided via glob)
    out_cnn = os.path.join(out_root, "cnn")
    out_tcn = os.path.join(out_root, "tcn")

    def run_and_maybe_foldavg(model_name: str, ckpt_spec: str, out_dir: str):
        ckpts = [ckpt_spec]
        if any(ch in ckpt_spec for ch in ["*", "?", "["]):
            ckpts = sorted(glob.glob(ckpt_spec))
        if len(ckpts) <= 1:
            infer_on_dir(input_dir=input_dir, output_dir=out_dir, checkpoint_path=ckpts[0] if ckpts else "", model_name=model_name)
            return
        # multiple checkpoints: write fold outputs to temp dirs and average into out_dir
        tmp_dirs = []
        for i, ck in enumerate(ckpts):
            td = os.path.join(out_dir, f"fold{i}")
            tmp_dirs.append(td)
            infer_on_dir(input_dir=input_dir, output_dir=td, checkpoint_path=ck, model_name=model_name)
        # average per file
        files = sorted(glob.glob(os.path.join(input_dir, "*.csv")))
        for p in files:
            base = os.path.splitext(os.path.basename(p))[0]
            mats = []
            for td in tmp_dirs:
                fp = os.path.join(td, f"{base}_probs.csv")
                if os.path.exists(fp):
                    mats.append(pd.read_csv(fp).values)
            if not mats:
                continue
            n = min(m.shape[0] for m in mats)
            stack = np.stack([m[:n] for m in mats], axis=0)
            avg = stack.mean(axis=0)
            os.makedirs(out_dir, exist_ok=True)
            pd.DataFrame(avg, columns=LABELS).to_csv(os.path.join(out_dir, f"{base}_probs.csv"), index=False)

    run_and_maybe_foldavg("cnn_bilstm", ckpt_cnn, out_cnn)
    run_and_maybe_foldavg("tcn", ckpt_tcn, out_tcn)

    # 2) Load weights
    w_cnn = 0.5
    if os.path.exists(weights_json):
        with open(weights_json, "r", encoding="utf-8") as f:
            w = json.load(f)
            if "cnn_bilstm" in w and "tcn" in w:
                w_cnn = float(w["cnn_bilstm"])
    w_tcn = 1.0 - w_cnn

    # 3) Average thresholds across folds and load optional calibrators
    thresholds = load_average_thresholds()
    calibrators = load_isotonic_calibrators()
    # window to frame rate approximation: one prob per window stride seconds
    stride_s = window_s * (1.0 - overlap)
    fps = 1.0 / stride_s if stride_s > 0 else 1.0

    # Load optional domain-specific postprocess config
    domain_name = "defog" if os.path.basename(os.path.dirname(input_dir)).lower().startswith("defog") or "defog" in input_dir.lower() else "tdcsfog"
    pp_cfg = None
    try:
        with open("configs/postprocess.yaml", "r", encoding="utf-8") as f:
            pp_all = yaml.safe_load(f)
            pp_cfg = pp_all.get("domains", {}).get(domain_name, {})
    except Exception:
        pp_cfg = None
    min_duration_s = float(pp_cfg.get("min_duration_s", 0.3)) if pp_cfg else 0.3
    merge_gap_s = float(pp_cfg.get("merge_gap_s", 0.5)) if pp_cfg else 0.5
    if pp_cfg and "thresholds" in pp_cfg:
        thresholds.update({k: float(v) for k, v in pp_cfg["thresholds"].items()})

    # 4) For each file, load probs, ensemble, save probs and intervals
    files = sorted(glob.glob(os.path.join(input_dir, "*.csv")))
    for p in files:
        base = os.path.splitext(os.path.basename(p))[0]
        pc = os.path.join(out_cnn, f"{base}_probs.csv")
        pt = os.path.join(out_tcn, f"{base}_probs.csv")
        if not (os.path.exists(pc) and os.path.exists(pt)):
            continue
        A = pd.read_csv(pc).values
        B = pd.read_csv(pt).values
        n = min(len(A), len(B))
        mix = w_cnn * A[:n] + w_tcn * B[:n]
        # Optional isotonic calibration per class
        if calibrators:
            for ci, c in enumerate(LABELS):
                cal = calibrators.get(c)
                if cal is not None:
                    try:
                        mix[:, ci] = np.clip(cal.predict(mix[:, ci]), 0.0, 1.0)
                    except Exception:
                        pass
        # Save mixed probs
        out_probs = os.path.join(out_root, f"{base}_probs_ens.csv")
        pd.DataFrame(mix, columns=LABELS).to_csv(out_probs, index=False)
        # Post-process to intervals
        intervals = probs_to_intervals(mix, LABELS, fps=fps, thresholds=thresholds, min_duration_s=min_duration_s, merge_gap_s=merge_gap_s)
        rows = []
        for c, lst in intervals.items():
            for (s, e, sc) in lst:
                rows.append({"class": c, "start_s": s, "end_s": e, "score": sc})
        out_int = os.path.join(out_root, f"{base}_intervals.csv")
        pd.DataFrame(rows).to_csv(out_int, index=False)
        logger.info({"file": base, "probs": out_probs, "intervals": out_int, "fps": fps, "w_cnn": w_cnn})


def main():
    # Example CLI usage could be added; for now, kept as library entrypoint
    pass


if __name__ == "__main__":
    main()


