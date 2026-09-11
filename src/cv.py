"""Experiment runner: frozen folds, nested TE per fold, cached OOF/test preds, results log."""
from __future__ import annotations

import hashlib
import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .config import EXP, FEATURES, N_FOLDS, TARGET
from .data import load_all
from .delong import paired
from .features.static import build_static
from .features.target_enc import TESpec, nested_te
from .folds import get_folds
from .models.gbdt import MODELS

_CACHE = {}


def get_data(version="v1"):
    if version not in _CACHE:
        tr, te, orig = load_all()
        static = build_static(tr, te, orig, version)
        y = tr[TARGET].to_numpy().astype(np.float64)
        _CACHE[version] = (tr, te, orig, static, y, get_folds(y))
    return _CACHE[version]


def signature(**kw) -> str:
    return hashlib.md5(json.dumps(kw, sort_keys=True, default=str).encode()).hexdigest()[:10]


def run(name: str, cols: list[str], te_specs: list[TESpec] | None = None, model="lgb", params=None,
        cat_cols=None, n_folds: int = N_FOLDS, folds_subset=None, static_version="v1", noise=False,
        ref: str | None = None, force=False, seed_te=0, extra_fn=None):
    """Train `model` on static `cols` + nested TE columns. Caches to experiments/<name>/.

    folds_subset: run only these fold ids (quick checks; pooled AUC then covers only those rows).
    ref: name of another experiment to compare against with paired DeLong.
    """
    te_specs = te_specs or []
    tr, te, orig, static, y, folds = get_data(static_version)
    assert n_folds == N_FOLDS, "folds are frozen at N_FOLDS"
    sig = signature(cols=cols, te=[(s.cols, s.m, s.decimals, s.binwidth, s.inner, s.offset, s.resid, s.modulus) for s in te_specs], model=model,
                    params=params, cat_cols=cat_cols, noise=noise, seed_te=seed_te, sv=static_version,
                    extra=getattr(extra_fn, "__name__", None))
    d = EXP / name
    meta_p = d / "meta.json"
    if meta_p.exists() and not force:
        meta = json.loads(meta_p.read_text())
        if meta["sig"] == sig and meta.get("folds_done") == list(range(n_folds)):
            print(f"[cached] {name}: OOF {meta['auc']:.6f}")
            return meta
        if meta["sig"] != sig:
            raise RuntimeError(f"experiment {name} exists with different signature; pick a new name or force=True")
    d.mkdir(exist_ok=True)

    ntr = len(tr)
    X_all = static[cols].copy()
    if noise:
        rng = np.random.default_rng(123)
        X_all["noise"] = rng.normal(size=len(X_all)).astype(np.float32)
    if extra_fn is not None:
        X_all = pd.concat([X_all, extra_fn(tr, te, static)], axis=1)
    X_tr_all, X_te_all = X_all.iloc[:ntr].reset_index(drop=True), X_all.iloc[ntr:].reset_index(drop=True)
    df_tr_raw, df_te_raw = tr[FEATURES], te[FEATURES]

    oof = np.full(ntr, np.nan)
    test_pred = np.zeros(len(te))
    fold_auc, iters = {}, {}
    fold_ids = list(range(n_folds)) if folds_subset is None else list(folds_subset)
    t0 = time.time()
    for k in fold_ids:
        tr_idx, va_idx = np.where(folds != k)[0], np.where(folds == k)[0]
        Xtr, Xva, Xte = X_tr_all.iloc[tr_idx].copy(), X_tr_all.iloc[va_idx].copy(), X_te_all.copy()
        p_base = None
        if any(s.resid for s in te_specs):
            # leak-free base probability: 1-D logistic fit of the recovered buy-score on this fold's training rows
            from sklearn.linear_model import LogisticRegression
            bs = static["buy_score"].to_numpy(np.float64).reshape(-1, 1)
            lr = LogisticRegression(C=1e6).fit(bs[:ntr][tr_idx], y[tr_idx])
            p_base = (lr.predict_proba(bs[:ntr])[:, 1], lr.predict_proba(bs[ntr:])[:, 1])
        for spec in te_specs:
            a, b, c = nested_te(spec, df_tr_raw, y, tr_idx, va_idx, df_te_raw, seed=seed_te, p_base=p_base)
            Xtr[spec.colname], Xva[spec.colname], Xte[spec.colname] = a, b, c
        pva, pte, it = MODELS[model](Xtr, y[tr_idx], Xva, y[va_idx], Xte, params=params, cat_cols=cat_cols)
        oof[va_idx] = pva
        test_pred += pte / len(fold_ids)
        fold_auc[k] = float(roc_auc_score(y[va_idx], pva))
        iters[k] = int(it)
        print(f"  {name} fold {k}: auc {fold_auc[k]:.6f} iters {it} ({time.time()-t0:.0f}s)", flush=True)
    mask = ~np.isnan(oof)
    auc = float(roc_auc_score(y[mask], oof[mask]))
    meta = dict(name=name, sig=sig, auc=auc, fold_auc=fold_auc, iters=iters, cols=list(X_tr_all.columns) +
                [s.colname for s in te_specs], model=model, params=params, folds_done=fold_ids,
                secs=round(time.time() - t0), time=time.strftime("%Y-%m-%d %H:%M"))
    if ref is not None and (EXP / ref / "oof.npy").exists():
        r = np.load(EXP / ref / "oof.npy")
        rm = mask & ~np.isnan(r)
        a_, b_, diff, se, z = paired(oof[rm], r[rm], y[rm])
        meta.update(ref=ref, ref_auc=b_, diff=diff, se=se, z=z)
        print(f"  vs {ref}: {diff:+.6f} (se {se:.6f}, z {z:+.1f})")
    np.save(d / "oof.npy", oof)
    np.save(d / "test.npy", test_pred)
    meta_p.write_text(json.dumps(meta, indent=1))
    with open(EXP / "results.csv", "a") as f:
        f.write(f"{meta['time']},{name},{model},{auc:.6f},{len(meta['cols'])},{len(fold_ids)},{meta.get('ref','')},{meta.get('diff', float('nan')):+.6f},{meta.get('z', float('nan')):+.2f}\n")
    print(f"== {name}: pooled OOF {auc:.6f} | folds {np.mean(list(fold_auc.values())):.6f} ± {np.std(list(fold_auc.values())):.6f}")
    return meta


def load_oof(name):
    return np.load(EXP / name / "oof.npy"), np.load(EXP / name / "test.npy")


def refit_full(name: str, cols, te_specs=None, model="lgb", params=None, static_version="v2", rounds_mult=1.1,
               seed_te=0, extra_fn=None):
    """Train on ALL train rows (no validation) with rounds = rounds_mult * mean best_iter of the CV run `name`.

    Test TE uses all of train; train-row TE uses inner folds. Saves experiments/<name>/test_refit.npy."""
    import lightgbm as lgb
    import xgboost as xgb
    from .models.gbdt import LGB_DEFAULT, XGB_DEFAULT
    te_specs = te_specs or []
    tr, te, orig, static, y, folds = get_data(static_version)
    meta = json.loads((EXP / name / "meta.json").read_text())
    n_rounds = int(round(rounds_mult * np.mean(list(meta["iters"].values()))))
    ntr = len(tr)
    X_all = static[cols].copy()
    if extra_fn is not None:
        X_all = pd.concat([X_all, extra_fn(tr, te, static)], axis=1)
    Xtr, Xte = X_all.iloc[:ntr].reset_index(drop=True).copy(), X_all.iloc[ntr:].reset_index(drop=True).copy()
    all_idx = np.arange(ntr)
    for spec in te_specs:
        a, _, c = nested_te(spec, tr[FEATURES], y, all_idx, all_idx[:1], te[FEATURES], seed=seed_te)
        Xtr[spec.colname], Xte[spec.colname] = a, c
    if model == "lgb":
        p = {**LGB_DEFAULT, **(params or {})}
        m = lgb.train(p, lgb.Dataset(Xtr, y), n_rounds)
        pred = m.predict(Xte)
    elif model == "xgb":
        p = {**XGB_DEFAULT, **(params or {})}
        m = xgb.train(p, xgb.DMatrix(Xtr, y), n_rounds)
        pred = m.predict(xgb.DMatrix(Xte))
    else:
        raise ValueError(model)
    np.save(EXP / name / "test_refit.npy", pred)
    print(f"refit {name}: {n_rounds} rounds on {ntr} rows")
    return pred
