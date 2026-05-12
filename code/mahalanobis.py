"""Mahalanobis open-set scorer with tied within-class covariance."""
from __future__ import annotations
from typing import Dict, Tuple
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def fit(features: np.ndarray, labels: np.ndarray,
        ridge: float = 1e-4) -> Tuple[Dict[int, np.ndarray], np.ndarray]:
    mus: Dict[int, np.ndarray] = {}
    centred_blocks = []
    for c in np.unique(labels):
        f = features[labels == c]
        mus[int(c)] = f.mean(axis=0)
        centred_blocks.append(f - mus[int(c)])
    centred = np.concatenate(centred_blocks, axis=0)
    Sigma = (centred.T @ centred) / centred.shape[0]
    Sigma += ridge * np.eye(Sigma.shape[0])
    return mus, np.linalg.inv(Sigma)


def score(features: np.ndarray,
          mus: Dict[int, np.ndarray],
          Sigma_inv: np.ndarray) -> np.ndarray:
    """Return min Mahalanobis distance over classes for each row."""
    dists = []
    for _, mu in mus.items():
        d = features - mu                              # (N, D)
        dists.append(np.einsum("nd,de,ne->n", d, Sigma_inv, d))
    return np.min(np.stack(dists, axis=1), axis=1)


def auroc(scores: np.ndarray, labels_ood: np.ndarray) -> float:
    """labels_ood: 1 = OOD, 0 = ID. Higher score = more OOD."""
    return float(roc_auc_score(labels_ood, scores))


def fpr_at_tpr(scores: np.ndarray, labels_ood: np.ndarray,
               tpr_target: float = 0.95) -> float:
    fpr, tpr, _ = roc_curve(labels_ood, scores)
    idx = int(np.searchsorted(tpr, tpr_target))
    idx = min(idx, len(fpr) - 1)
    return float(fpr[idx])
