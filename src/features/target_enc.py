"""Nested out-of-fold target encoding. Keys are int64 arrays built from one or more columns."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def make_key(df: pd.DataFrame, cols, decimals=1) -> np.ndarray:
    key = np.zeros(len(df), dtype=np.int64)
    for c in cols:
        v = np.round(df[c].to_numpy(dtype=float) * 10**decimals).astype(np.int64)
        key = key * 10_000_000 + v
    return key


def te_fit_apply(key_fit, y_fit, key_apply, m: float, prior: float):
    u, inv = np.unique(key_fit, return_inverse=True)
    s = np.bincount(inv, weights=y_fit, minlength=len(u))
    c = np.bincount(inv, minlength=len(u)).astype(float)
    enc = (s + m * prior) / (c + m)
    idx = np.searchsorted(u, key_apply)
    idx_c = np.minimum(idx, len(u) - 1)
    found = u[idx_c] == key_apply
    return np.where(found, enc[idx_c], prior).astype(np.float32)


@dataclass
class TESpec:
    cols: tuple
    m: float = 10.0
    decimals: int = 1
    binwidth: float | None = None  # if set, first column is binned by this width
    name: str = ""
    inner: int = 5  # inner folds for the training-row encoding
    offset: float = 0.0  # shift applied before binning (half-width offset gives staggered bins)
    resid: bool = False  # encode mean(y - p_base) instead of mean(y); p_base supplied by the runner

    def key(self, df: pd.DataFrame) -> np.ndarray:
        if self.binwidth is None:
            return make_key(df, self.cols, self.decimals)
        d = df[list(self.cols)].copy()
        d[self.cols[0]] = np.floor((d[self.cols[0]] + self.offset) / self.binwidth)
        return make_key(d, self.cols, 0)

    @property
    def colname(self):
        if self.name:
            return self.name
        b = f"_b{self.binwidth:g}" if self.binwidth else ""
        i = f"_i{self.inner}" if self.inner != 5 else ""
        i += f"_o{self.offset:g}" if self.offset else ""
        i += "_r" if self.resid else ""
        return "te_" + "_".join(c[:6] for c in self.cols) + b + f"_m{self.m:g}" + i


def nested_te(spec: TESpec, df_tr: pd.DataFrame, y: np.ndarray, tr_idx, va_idx, df_te: pd.DataFrame,
              inner_folds=None, seed=0, p_base=None):
    """Returns (enc_train_rows[tr_idx order], enc_valid_rows, enc_test).

    p_base: optional (train_probs, test_probs) used when spec.resid is set; the encoded target becomes y - p_base."""
    inner_folds = inner_folds or spec.inner
    k_all = spec.key(df_tr)
    k_te = spec.key(df_te)
    y_strat = y[tr_idx]
    if spec.resid:
        assert p_base is not None, "resid TE needs p_base"
        y = y - p_base[0]
    prior = float(y[tr_idx].mean())
    k_tr, y_tr = k_all[tr_idx], y[tr_idx]
    enc_tr = np.empty(len(tr_idx), dtype=np.float32)
    skf = StratifiedKFold(inner_folds, shuffle=True, random_state=seed)
    for a, b in skf.split(np.zeros(len(tr_idx)), y_strat):
        enc_tr[b] = te_fit_apply(k_tr[a], y_tr[a], k_tr[b], spec.m, prior)
    enc_va = te_fit_apply(k_tr, y_tr, k_all[va_idx], spec.m, prior)
    enc_te = te_fit_apply(k_tr, y_tr, k_te, spec.m, prior)
    return enc_tr, enc_va, enc_te
