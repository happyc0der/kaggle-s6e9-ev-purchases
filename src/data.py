"""Load raw competition + original data, encode categoricals to fixed ints, cache as parquet."""
from __future__ import annotations

import pandas as pd

from .config import CAT_MAPS, CAT_COLS, ID, ORIG_FILE, PROC, RAW, TARGET


def _encode(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in CAT_COLS:
        if c in df:
            m = CAT_MAPS[c]
            bad = set(df[c].dropna().unique()) - set(m)
            if bad:
                raise ValueError(f"unexpected values in {c}: {bad}")
            df[c] = df[c].map(m).astype("int8")
    if TARGET in df:
        df[TARGET] = df[TARGET].map({"No": 0, "Yes": 1}).astype("int8")
    return df


def load(kind: str) -> pd.DataFrame:
    """kind in {'train','test','original'}; cached parquet after first read."""
    cache = PROC / f"{kind}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    if kind == "original":
        df = pd.read_csv(RAW / ORIG_FILE)
        df = df.rename(columns={"Buyer_ID": ID})
    else:
        df = pd.read_csv(RAW / f"{kind}.csv")
    df = _encode(df)
    df.to_parquet(cache)
    return df


def load_all():
    tr, te, orig = load("train"), load("test"), load("original")
    return tr, te, orig
