"""Similarity of each competition row to the original rows that share its (income) value.

Target-free w.r.t. the competition label; uses original labels only (public data)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import FEATURES

INC, COM = "Annual_Income_USD", "Daily_Commute_km"


def _best_match(df: pd.DataFrame, orig: pd.DataFrame, key_cols, other_cols, prefix: str) -> pd.DataFrame:
    n = len(df)
    best = np.full(n, -1, dtype=np.int8)
    mean_m = np.full(n, -1.0, dtype=np.float32)
    lab_best = np.full(n, -1.0, dtype=np.float32)
    lab_mean = np.full(n, -1.0, dtype=np.float32)
    n_orig = np.zeros(n, dtype=np.int16)
    o = orig.dropna(subset=list(key_cols))
    okey = pd.MultiIndex.from_frame(o[list(key_cols)]) if len(key_cols) > 1 else pd.Index(o[key_cols[0]])
    dkey = pd.MultiIndex.from_frame(df[list(key_cols)]) if len(key_cols) > 1 else pd.Index(df[key_cols[0]])
    ogroups = o.groupby(list(key_cols)).indices
    oX = o[other_cols].to_numpy(dtype=float)
    oy = o["Will_Buy_EV"].to_numpy(dtype=float)
    dX = df[other_cols].to_numpy(dtype=float)
    dgroups = df.groupby(list(key_cols)).indices
    for k, didx in dgroups.items():
        oidx = ogroups.get(k)
        if oidx is None:
            continue
        A = dX[didx][:, None, :]  # (nd,1,c)
        B = oX[oidx][None, :, :]  # (1,no,c)
        M = np.nan_to_num((A == B).astype(np.float32), nan=0.0).sum(2)  # (nd,no) matches; NaN in orig never matches
        bm = M.max(1)
        best[didx] = bm
        mean_m[didx] = M.mean(1)
        n_orig[didx] = len(oidx)
        is_best = M == bm[:, None]
        lab_best[didx] = (is_best * oy[oidx][None, :]).sum(1) / is_best.sum(1)
        lab_mean[didx] = oy[oidx].mean()
    return pd.DataFrame({
        f"{prefix}_bestmatch": best, f"{prefix}_meanmatch": mean_m, f"{prefix}_labbest": lab_best,
        f"{prefix}_labmean": lab_mean, f"{prefix}_norig": n_orig,
    })


def build_origmatch(df: pd.DataFrame, orig: pd.DataFrame) -> pd.DataFrame:
    oth_inc = [c for c in FEATURES if c != INC]
    oth_com = [c for c in FEATURES if c != COM]
    oth_pair = [c for c in FEATURES if c not in (INC, COM)]
    parts = [
        _best_match(df, orig, (INC,), oth_inc, "om_inc"),
        _best_match(df, orig, (COM,), oth_com, "om_com"),
        _best_match(df, orig, (INC, COM), oth_pair, "om_pair"),
    ]
    return pd.concat(parts, axis=1)


GROUP = [f"om_{k}_{s}" for k in ("inc", "com", "pair") for s in ("bestmatch", "meanmatch", "labbest", "labmean", "norig")]
