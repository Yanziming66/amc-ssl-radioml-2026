"""Split conformal Adaptive Prediction Sets (APS) — Romano et al. 2020."""
from __future__ import annotations
from typing import List, Set, Tuple
import numpy as np


def aps_calibrate(softmax_cal: np.ndarray,
                  y_cal: np.ndarray,
                  alpha: float = 0.1) -> float:
    """Return the calibration quantile q̂_α."""
    order = np.argsort(-softmax_cal, axis=1)                     # descending
    sorted_p = np.take_along_axis(softmax_cal, order, axis=1)
    cum = np.cumsum(sorted_p, axis=1)
    rank_y = np.argsort(order, axis=1)
    rank_of_true = rank_y[np.arange(len(y_cal)), y_cal]          # (N,)
    scores = cum[np.arange(len(y_cal)), rank_of_true]            # E(x_i, y_i)
    n = len(y_cal)
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    q_level = min(q_level, 1.0)
    return float(np.quantile(scores, q_level, method="higher"))


def aps_predict(softmax_test: np.ndarray, q: float) -> List[Set[int]]:
    order = np.argsort(-softmax_test, axis=1)
    sorted_p = np.take_along_axis(softmax_test, order, axis=1)
    cum = np.cumsum(sorted_p, axis=1)
    sets: List[Set[int]] = []
    for i in range(softmax_test.shape[0]):
        # smallest k such that cum[i, k-1] >= q
        k = int(np.searchsorted(cum[i], q)) + 1
        k = min(k, softmax_test.shape[1])
        sets.append(set(order[i, :k].tolist()))
    return sets


def coverage_and_size(sets: List[Set[int]],
                      y_test: np.ndarray) -> Tuple[float, float]:
    cov = float(np.mean([y in s for s, y in zip(sets, y_test)]))
    size = float(np.mean([len(s) for s in sets]))
    return cov, size
