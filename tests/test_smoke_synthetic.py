"""End-to-end smoke test on a synthetic stand-in with the same schema (no real data needed)."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.config import FEATURES, CAT_MAPS
from src.features.static import GROUPS, base_extras, density, digits, pair_counts, value_stats, PAIRS_DEFAULT
from src.features.target_enc import TESpec, nested_te
from src.models.gbdt import fit_lgb, fit_xgb, fit_cb


def synth(n, seed):
    r = np.random.RandomState(seed)
    df = pd.DataFrame({
        "Age": r.randint(25, 70, n),
        "Gender": r.choice(3, n, p=[.52, .45, .03]),
        "Annual_Income_USD": np.round(np.clip(r.normal(85000, 35000, n), 30000, None), 1),
        "City_Type": r.choice(3, n),
        "Daily_Commute_km": np.round(np.clip(r.normal(40, 25, n), 5, None), 1),
        "Number_of_Cars_Owned": r.randint(1, 5, n),
        "Current_Car_Type": r.choice(4, n),
        "Charging_Stations_Near_Home": r.randint(0, 15, n),
        "Charging_Stations_Near_Work": r.randint(0, 20, n),
        "Home_Charging_Possible": r.randint(0, 2, n),
        "Environmental_Concern_Level": r.randint(1, 6, n),
        "Subsidy_Available": r.randint(0, 2, n),
        "Range_Anxiety_Level": r.choice(3, n),
    })
    score = 1.2 * df.Annual_Income_USD / 1e5 + 0.6 * df.Environmental_Concern_Level + 2 * df.Subsidy_Available \
        - 1 * (df.Range_Anxiety_Level == 1) - 3 * (df.Range_Anxiety_Level == 2) + r.normal(size=n)
    y = (score > 5.5).astype(float).to_numpy()
    return df[FEATURES], y


def test_pipeline():
    tr, y = synth(20000, 0)
    te, _ = synth(5000, 1)
    orig, _ = synth(2000, 2)
    df = pd.concat([tr, te], ignore_index=True)
    parts = [df, base_extras(df),
             value_stats(df.Annual_Income_USD.to_numpy(float), orig.Annual_Income_USD.to_numpy(float), 1, "inc"),
             value_stats(df.Daily_Commute_km.to_numpy(float), orig.Daily_Commute_km.to_numpy(float), 1, "com"),
             density(df.Annual_Income_USD.to_numpy(float), [50, 250, 1000], "inc"),
             density(df.Daily_Commute_km.to_numpy(float), [0.1, 0.5, 2], "com"),
             digits(df), pair_counts(df, PAIRS_DEFAULT)]
    static = pd.concat(parts, axis=1)
    for g, cols in GROUPS.items():
        assert all(c in static for c in cols), g
    assert static.isna().sum().sum() == 0
    cols = GROUPS["raw"] + GROUPS["inc_stats"] + GROUPS["digits"]
    Xtr_all, Xte = static.iloc[:len(tr)][cols], static.iloc[len(tr):][cols].reset_index(drop=True)
    idx = np.arange(len(tr)); tr_idx, va_idx = idx[:16000], idx[16000:]
    Xtr, Xva = Xtr_all.iloc[tr_idx].copy(), Xtr_all.iloc[va_idx].copy()
    spec = TESpec(("Annual_Income_USD",), m=10, binwidth=1000)
    a, b, c = nested_te(spec, tr, y, tr_idx, va_idx, te)
    Xtr[spec.colname], Xva[spec.colname], Xte[spec.colname] = a, b, c
    for fn, kw in [(fit_lgb, {}), (fit_xgb, {}), (fit_cb, {"cat_cols": ["Annual_Income_USD"]})]:
        pva, pte, it = fn(Xtr, y[tr_idx], Xva, y[va_idx], Xte, params={"learning_rate": 0.1}, rounds=300, es=30, **kw)
        auc = roc_auc_score(y[va_idx], pva)
        assert auc > 0.85, (fn.__name__, auc)
        assert len(pte) == len(te) and it >= 0
