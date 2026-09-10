"""Fast DeLong AUC variance / paired comparison (Sun & Xu 2014)."""
from __future__ import annotations

import numpy as np
from scipy import stats


def _midrank(x: np.ndarray) -> np.ndarray:
    return stats.rankdata(x, method="average")


def delong(preds: np.ndarray, y: np.ndarray):
    """preds: (k, n) score vectors on the same rows; y: 0/1. Returns (auc[k], cov[k,k])."""
    preds = np.atleast_2d(np.asarray(preds, dtype=np.float64))
    y = np.asarray(y).astype(bool)
    pos, neg = preds[:, y], preds[:, ~y]
    m, n, k = pos.shape[1], neg.shape[1], preds.shape[0]
    tx = np.array([_midrank(pos[r]) for r in range(k)])
    ty = np.array([_midrank(neg[r]) for r in range(k)])
    tz = np.array([_midrank(np.concatenate([pos[r], neg[r]])) for r in range(k)])
    auc = (tz[:, :m].sum(1) / m - (m + 1) / 2) / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1 - (tz[:, m:] - ty) / m
    cov = np.atleast_2d(np.cov(v01)) / m + np.atleast_2d(np.cov(v10)) / n
    return auc, cov


def paired(a: np.ndarray, b: np.ndarray, y: np.ndarray):
    """Return (auc_a, auc_b, diff, se_diff, z) for two OOF vectors on the same rows."""
    auc, cov = delong(np.vstack([a, b]), y)
    se = float(np.sqrt(max(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1], 1e-30)))
    d = float(auc[0] - auc[1])
    return float(auc[0]), float(auc[1]), d, se, d / se
