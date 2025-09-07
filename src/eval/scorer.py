from typing import List, Dict, Tuple
import numpy as np
from .postprocess import iou_1d


def average_precision(rec: np.ndarray, prec: np.ndarray) -> float:
    # monotonic interpolation
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return float(ap)


def score_intervals(preds: Dict[str, List[Tuple[float, float, float]]],
                    gts: Dict[str, List[Tuple[float, float]]],
                    classes: List[str], iou_thr: float = 0.5) -> Dict[str, float]:
    ap_by_class = {}
    for c in classes:
        p = sorted(preds.get(c, []), key=lambda x: -x[2])
        gt = gts.get(c, [])
        npos = len(gt)
        if npos == 0:
            ap_by_class[c] = 0.0
            continue
        matched = np.zeros(npos, dtype=bool)
        tp = np.zeros(len(p))
        fp = np.zeros(len(p))
        for i, (ps, pe, sc) in enumerate(p):
            ious = np.array([iou_1d((ps, pe), g) for g in gt])
            j = int(np.argmax(ious)) if ious.size > 0 else -1
            if j >= 0 and ious[j] >= iou_thr and not matched[j]:
                tp[i] = 1
                matched[j] = True
            else:
                fp[i] = 1
        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        rec = cum_tp / max(npos, 1)
        prec = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)
        ap_by_class[c] = average_precision(rec, prec)
    return ap_by_class


def mean_ap(ap_by_class: Dict[str, float]) -> float:
    vals = list(ap_by_class.values())
    return float(np.mean(vals)) if vals else 0.0
