from typing import List, Dict, Tuple
import numpy as np


def probs_to_intervals(probs: np.ndarray, classes: List[str], fps: float,
                       thresholds: Dict[str, float], hysteresis: Tuple[float, float] = (0.6, 0.4),
                       min_duration_s: float = 0.3, merge_gap_s: float = 0.5,
                       smooth: int = 1) -> Dict[str, List[Tuple[float, float, float]]]:
    """
    probs: (T, K) array, per-frame probabilities
    returns per-class list of (start_s, end_s, score)
    """
    start_thr, cont_thr = hysteresis
    results = {c: [] for c in classes}
    T, K = probs.shape
    for ci, c in enumerate(classes):
        p = probs[:, ci]
        if smooth and smooth > 1 and p.size:
            k = np.ones(int(smooth), dtype=np.float32) / float(smooth)
            # same-length convolution with edge handling
            pad = smooth // 2
            p_pad = np.pad(p, (pad, pad), mode='edge')
            p = np.convolve(p_pad, k, mode='valid')
        on = False
        start = 0
        scores = []
        thr = thresholds.get(c, start_thr)
        for t in range(T):
            if not on and p[t] >= thr:
                on = True
                start = t
                scores = [p[t]]
            elif on:
                scores.append(p[t])
                if p[t] < cont_thr:
                    end = t
                    dur = (end - start) / fps
                    if dur >= min_duration_s:
                        results[c].append((start / fps, end / fps, float(np.max(scores))))
                    on = False
        if on:
            end = T - 1
            dur = (end - start) / fps
            if dur >= min_duration_s:
                results[c].append((start / fps, end / fps, float(np.max(scores))))
        # merge gaps
        merged = []
        for (s, e, sc) in sorted(results[c]):
            if not merged:
                merged.append([s, e, sc])
            else:
                ps, pe, psc = merged[-1]
                if s - pe <= merge_gap_s:
                    merged[-1][1] = e
                    merged[-1][2] = max(psc, sc)
                else:
                    merged.append([s, e, sc])
        results[c] = [(s, e, sc) for s, e, sc in merged]
    return results


def iou_1d(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    s1, e1 = a
    s2, e2 = b
    inter = max(0.0, min(e1, e2) - max(s1, s2))
    union = max(e1, e2) - min(s1, s2)
    return inter / union if union > 0 else 0.0
