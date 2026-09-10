"""Target-free features computed once on train+test (+original). Cached by version tag."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import FEATURES, PROC

INC, COM = "Annual_Income_USD", "Daily_Commute_km"


def _key(a: np.ndarray, decimals: int) -> np.ndarray:
    return np.round(a * 10**decimals).astype(np.int64)


def value_stats(all_vals: np.ndarray, orig_vals: np.ndarray, decimals: int, prefix: str) -> pd.DataFrame:
    """Per-value frequency in train+test, in original, lift, novelty, nearest-original distance."""
    k = _key(all_vals, decimals)
    ko = _key(orig_vals[~np.isnan(orig_vals)], decimals)
    u, inv, cnt = np.unique(k, return_inverse=True, return_counts=True)
    uo, cnto = np.unique(ko, return_counts=True)
    idx = np.searchsorted(uo, u)
    idx_c = np.minimum(idx, len(uo) - 1)
    found = uo[idx_c] == u
    cnt_orig_u = np.where(found, cnto[idx_c], 0)
    share_all = cnt / len(k)
    share_orig = cnt_orig_u / len(ko)
    lift_u = share_all / np.where(share_orig > 0, share_orig, 1 / len(ko))  # novel values: vs 1 pseudo-count
    # nearest original value distance (signed)
    uo_f = uo / 10**decimals
    u_f = u / 10**decimals
    pos = np.searchsorted(uo_f, u_f)
    lo = uo_f[np.clip(pos - 1, 0, len(uo_f) - 1)]
    hi = uo_f[np.clip(pos, 0, len(uo_f) - 1)]
    d_lo, d_hi = u_f - lo, hi - u_f
    dist_u = np.where(d_lo <= d_hi, -d_lo, d_hi)
    dist_u[found] = 0.0
    rank_u = cnt.argsort().argsort() / len(cnt)
    out = pd.DataFrame(
        {
            f"{prefix}_cnt": cnt[inv].astype(np.int32),
            f"{prefix}_cnt_orig": cnt_orig_u[inv].astype(np.int16),
            f"{prefix}_lift": np.log1p(lift_u[inv]).astype(np.float32),
            f"{prefix}_novel": (~found)[inv].astype(np.int8),
            f"{prefix}_dist_orig": dist_u[inv].astype(np.float32),
            f"{prefix}_freqrank": rank_u[inv].astype(np.float32),
        }
    )
    return out


def density(all_vals: np.ndarray, windows, prefix: str) -> pd.DataFrame:
    s = np.sort(all_vals)
    out = {}
    for w in windows:
        c = np.searchsorted(s, all_vals + w, side="right") - np.searchsorted(s, all_vals - w, side="left")
        out[f"{prefix}_dens{w:g}"] = c.astype(np.int32)
    return pd.DataFrame(out)


def digits(df: pd.DataFrame) -> pd.DataFrame:
    inc10 = np.round(df[INC].to_numpy() * 10).astype(np.int64)  # tenths of a dollar
    com10 = np.round(df[COM].to_numpy() * 10).astype(np.int64)
    out = {
        "inc_d_dec": inc10 % 10,
        "inc_d1": (inc10 // 10) % 10,
        "inc_d2": (inc10 // 100) % 10,
        "inc_d3": (inc10 // 1000) % 10,
        "inc_d4": (inc10 // 10000) % 10,
        "inc_mod100": ((inc10 % 1000) == 0).astype(np.int8),
        "inc_mod1000": ((inc10 % 10000) == 0).astype(np.int8),
        "inc_is_int": ((inc10 % 10) == 0).astype(np.int8),
        "inc_is_30k": (inc10 == 300000).astype(np.int8),
        "com_d_dec": com10 % 10,
        "com_d1": (com10 // 10) % 10,
        "com_is_int": ((com10 % 10) == 0).astype(np.int8),
        "com_is_5": (com10 == 50).astype(np.int8),
    }
    return pd.DataFrame({k: np.asarray(v).astype(np.int8) for k, v in out.items()})


def pair_counts(df: pd.DataFrame, pairs) -> pd.DataFrame:
    out = {}
    for a, b in pairs:
        ka = _key(df[a].to_numpy(), 1)
        kb = _key(df[b].to_numpy(), 1)
        key = ka * 10_000_000 + kb
        _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
        out[f"cnt_{a[:6]}_{b[:6]}"] = cnt[inv].astype(np.int32)
    return pd.DataFrame(out)


def base_extras(df: pd.DataFrame) -> pd.DataFrame:
    """Cheap domain helpers (known to be near-noise, kept for completeness)."""
    out = pd.DataFrame(index=df.index)
    out["worry"] = (df["Range_Anxiety_Level"].map({0: 0, 1: -1, 2: -3})).astype(np.int8)
    out["buy_score"] = (
        1.2 * df[INC] / 1e5
        + 0.6 * df["Environmental_Concern_Level"]
        + 2 * df["Subsidy_Available"]
        + out["worry"]
    ).astype(np.float32)
    out["chargers_total"] = (df["Charging_Stations_Near_Home"] + df["Charging_Stations_Near_Work"]).astype(np.int8)
    out["inc_x_sub"] = (df[INC] * df["Subsidy_Available"]).astype(np.float32)
    out["con_x_sub"] = (df["Environmental_Concern_Level"] * df["Subsidy_Available"]).astype(np.int8)
    return out


PAIRS_DEFAULT = [
    (INC, COM), (INC, "Age"), (INC, "City_Type"), (INC, "Subsidy_Available"),
    (INC, "Environmental_Concern_Level"), (INC, "Range_Anxiety_Level"),
    (COM, "Age"), (COM, "City_Type"), (COM, "Subsidy_Available"),
    (COM, "Environmental_Concern_Level"), (COM, "Range_Anxiety_Level"),
]


def build_static(tr: pd.DataFrame, te: pd.DataFrame, orig: pd.DataFrame, version: str = "v1") -> pd.DataFrame:
    """Return one frame with len(tr)+len(te) rows (train first), all target-free columns."""
    cache = PROC / f"static_{version}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    if version == "v2":  # v1 + original-row similarity features
        from .origmatch import build_origmatch
        base = build_static(tr, te, orig, "v1")
        df = pd.concat([tr[FEATURES], te[FEATURES]], ignore_index=True)
        out = pd.concat([base, build_origmatch(df, orig)], axis=1)
        out.to_parquet(cache)
        return out
    df = pd.concat([tr[FEATURES], te[FEATURES]], ignore_index=True)
    parts = [df.reset_index(drop=True)]
    parts.append(base_extras(df))
    parts.append(value_stats(df[INC].to_numpy(float), orig[INC].to_numpy(float), 1, "inc"))
    parts.append(value_stats(df[COM].to_numpy(float), orig[COM].to_numpy(float), 1, "com"))
    parts.append(density(df[INC].to_numpy(float), [50, 250, 1000], "inc"))
    parts.append(density(df[COM].to_numpy(float), [0.1, 0.5, 2], "com"))
    parts.append(digits(df))
    parts.append(pair_counts(df, PAIRS_DEFAULT))
    out = pd.concat(parts, axis=1)
    out.to_parquet(cache)
    return out


# named column groups for ablations
GROUPS = {
    "raw": FEATURES,
    "extras": ["worry", "buy_score", "chargers_total", "inc_x_sub", "con_x_sub"],
    "inc_stats": ["inc_cnt", "inc_cnt_orig", "inc_lift", "inc_novel", "inc_dist_orig", "inc_freqrank"],
    "com_stats": ["com_cnt", "com_cnt_orig", "com_lift", "com_novel", "com_dist_orig", "com_freqrank"],
    "density": ["inc_dens50", "inc_dens250", "inc_dens1000", "com_dens0.1", "com_dens0.5", "com_dens2"],
    "digits": ["inc_d_dec", "inc_d1", "inc_d2", "inc_d3", "inc_d4", "inc_mod100", "inc_mod1000", "inc_is_int",
               "inc_is_30k", "com_d_dec", "com_d1", "com_is_int", "com_is_5"],
    "pairs": [f"cnt_{a[:6]}_{b[:6]}" for a, b in PAIRS_DEFAULT],
    "origmatch": [f"om_{k}_{s}" for k in ("inc", "com", "pair") for s in ("bestmatch", "meanmatch", "labbest", "labmean", "norig")],
}
