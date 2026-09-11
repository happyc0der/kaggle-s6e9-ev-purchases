"""Uniform fit/predict wrappers. Each returns (pred_valid, pred_test, best_iter)."""
from __future__ import annotations

import numpy as np

LGB_DEFAULT = dict(
    objective="binary", metric="auc", learning_rate=0.02, num_leaves=63, min_child_samples=100,
    feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, max_bin=255,
    verbose=-1, num_threads=12, seed=0,
)
XGB_DEFAULT = dict(
    objective="binary:logistic", eval_metric="auc", learning_rate=0.02, max_depth=7, min_child_weight=10,
    subsample=0.8, colsample_bytree=0.7, reg_lambda=1.0, tree_method="hist", max_bin=256, nthread=12, seed=0,
)
CB_DEFAULT = dict(
    loss_function="Logloss", eval_metric="AUC", learning_rate=0.05, depth=7, l2_leaf_reg=3, border_count=254,
    thread_count=12, random_seed=0, verbose=0,
)


def fit_lgb(Xtr, ytr, Xva, yva, Xte, params=None, rounds=20000, es=300, cat_cols=None):
    import lightgbm as lgb

    p = {**LGB_DEFAULT, **(params or {})}
    fixed = p.pop("_fixed_rounds", None)
    if cat_cols:
        Xtr, Xva, Xte = (x.copy() for x in (Xtr, Xva, Xte))
        for c in cat_cols:
            for x in (Xtr, Xva, Xte):
                x[c] = np.round(x[c].to_numpy(float) * 10).astype(np.int64)
    dtr = lgb.Dataset(Xtr, ytr, categorical_feature=cat_cols or "auto", free_raw_data=False)
    if fixed:
        m = lgb.train(p, dtr, fixed)
        return m.predict(Xva), m.predict(Xte), fixed
    dva = lgb.Dataset(Xva, yva, reference=dtr, categorical_feature=cat_cols or "auto")
    m = lgb.train(p, dtr, rounds, valid_sets=[dva], callbacks=[lgb.early_stopping(es, verbose=False)])
    return m.predict(Xva, num_iteration=m.best_iteration), m.predict(Xte, num_iteration=m.best_iteration), m.best_iteration


def fit_xgb(Xtr, ytr, Xva, yva, Xte, params=None, rounds=20000, es=300, cat_cols=None):
    import xgboost as xgb

    p = {**XGB_DEFAULT, **(params or {})}
    fixed = p.pop("_fixed_rounds", None)
    dtr = xgb.DMatrix(Xtr, ytr)
    dva = xgb.DMatrix(Xva, yva)
    dte = xgb.DMatrix(Xte)
    if fixed:
        m = xgb.train(p, dtr, fixed)
        return m.predict(dva), m.predict(dte), fixed
    m = xgb.train(p, dtr, rounds, evals=[(dva, "va")], early_stopping_rounds=es, verbose_eval=False)
    it = m.best_iteration + 1
    return m.predict(dva, iteration_range=(0, it)), m.predict(dte, iteration_range=(0, it)), m.best_iteration


def fit_cb(Xtr, ytr, Xva, yva, Xte, params=None, rounds=20000, es=300, cat_cols=None):
    from catboost import CatBoostClassifier, Pool

    p = {**CB_DEFAULT, **(params or {})}
    fixed = p.pop("_fixed_rounds", None)
    cats = [Xtr.columns.get_loc(c) for c in (cat_cols or [])]
    if cats:  # catboost needs int/str categoricals
        Xtr, Xva, Xte = (x.copy() for x in (Xtr, Xva, Xte))
        for c in cat_cols:
            for x in (Xtr, Xva, Xte):
                x[c] = np.round(x[c].to_numpy(float) * 10).astype(np.int64)
    ptr, pva, pte = Pool(Xtr, ytr, cat_features=cats), Pool(Xva, yva, cat_features=cats), Pool(Xte, cat_features=cats)
    if fixed:
        m = CatBoostClassifier(iterations=fixed, **p)
        m.fit(ptr)
        return m.predict_proba(pva)[:, 1], m.predict_proba(pte)[:, 1], fixed
    m = CatBoostClassifier(iterations=rounds, od_type="Iter", od_wait=es, **p)
    m.fit(ptr, eval_set=pva, use_best_model=True)
    return m.predict_proba(pva)[:, 1], m.predict_proba(pte)[:, 1], m.get_best_iteration()


MODELS = {"lgb": fit_lgb, "xgb": fit_xgb, "cb": fit_cb}


def _fit_nn(*a, **k):
    from .nn import fit_nn
    return fit_nn(*a, **k)


MODELS["nn"] = _fit_nn
