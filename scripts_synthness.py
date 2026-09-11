"""(1) Adversarial 'synthetic-ness' score: P(row is synthetic | x) learned jointly over all columns vs the original 10k.
   Target-free w.r.t. the competition label. (2) Residual probe: can a fresh model predict y - p_blend from features?"""
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score
from scipy.special import logit, expit
from src.cv import get_data, load_oof, run
from src.config import PROC
from src.config import FEATURES
from src.features.static import GROUPS
from src.features.target_enc import TESpec
tr, te, orig, static, y, folds = get_data("v2")
INC, COM = "Annual_Income_USD", "Daily_Commute_km"
# ---- (1) synthness
n = len(tr)
o = orig[FEATURES].copy()
X_all = pd.concat([static[FEATURES], o], ignore_index=True)
z = np.r_[np.ones(len(static)), np.zeros(len(o))]
syn = np.zeros(len(X_all))
skf = StratifiedKFold(5, shuffle=True, random_state=0)
P = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_child_samples=50, feature_fraction=0.8, verbose=-1, num_threads=12,
         scale_pos_weight=1.0, is_unbalance=False)
for a, b in skf.split(X_all, z):
    m = lgb.train(P, lgb.Dataset(X_all.iloc[a], z[a]), 400)
    syn[b] = m.predict(X_all.iloc[b])
print("synthness AUC (synthetic vs original):", round(roc_auc_score(z, syn), 4))
syn_static = syn[:len(static)]
np.save(PROC / "synthness.npy", syn_static)
print("corr(synthness, y) on train:", round(np.corrcoef(syn_static[:n], y)[0, 1], 4))
# ---- (2) residual probe on the blend
lg, _ = load_oof("lgb_final_bag8"); cb, _ = load_oof("cb_final_bag6")
p = expit(0.4 * logit(np.clip(lg, 1e-6, 1 - 1e-6)) + 0.6 * logit(np.clip(cb, 1e-6, 1 - 1e-6)))
r = y - p
Xr = static.iloc[:n][FEATURES].copy(); Xr["p"] = p; Xr["syn"] = syn_static[:n]
Xr["inc_cat"] = np.round(Xr[INC]).astype(np.int64)
pred_r = np.zeros(n)
for a, b in KFold(5, shuffle=True, random_state=0).split(Xr):
    m = lgb.train(dict(objective="regression", learning_rate=0.05, num_leaves=63, min_child_samples=200, feature_fraction=0.8, verbose=-1, num_threads=12),
                  lgb.Dataset(Xr.iloc[a], r[a]), 300)
    pred_r[b] = m.predict(Xr.iloc[b])
ss_res = np.sum((r - pred_r) ** 2); ss_tot = np.sum((r - r.mean()) ** 2)
print("residual probe R^2:", round(1 - ss_res / ss_tot, 5))
p2 = np.clip(p + pred_r, 1e-6, 1 - 1e-6)
print("AUC blend:", round(roc_auc_score(y, p), 6), " AUC blend + residual model:", round(roc_auc_score(y, p2), 6))
# ---- (3) synthness as a feature in the shallow recipe
full = GROUPS["raw"] + GROUPS["inc_stats"] + GROUPS["com_stats"] + GROUPS["digits"]
def temulti(m=3):
    return [TESpec((INC,), m=m), TESpec((COM,), m=m), TESpec((INC,), m=m, binwidth=100), TESpec((INC,), m=m, binwidth=500),
            TESpec((INC,), m=m, binwidth=2000), TESpec((COM,), m=m, binwidth=1), TESpec((COM,), m=m, binwidth=5)]
OTHER = ["Age", "Charging_Stations_Near_Home", "Charging_Stations_Near_Work", "Environmental_Concern_Level", "Number_of_Cars_Owned",
         "Gender", "City_Type", "Current_Car_Type", "Home_Charging_Possible", "Subsidy_Available", "Range_Anxiety_Level"]
DIG = ["inc_d1", "inc_d2", "inc_d3", "inc_d4", "com_d_dec", "com_d1"]
def te_all(ms=(10, 100)):
    specs = []
    for m in ms:
        specs += [TESpec((c,), m=m) for c in OTHER + DIG] + [TESpec((INC,), m=m, binwidth=1000)]
    return specs
S = {"max_depth": 5, "num_leaves": 32, "min_child_samples": 10, "feature_fraction": 0.3, "bagging_fraction": 0.8,
     "lambda_l1": 0.071, "lambda_l2": 2.0, "max_bin": 1024}
def syn_fn(tr, te, static):
    s = np.load(PROC / "synthness.npy")
    return pd.DataFrame({"synthness": s.astype(np.float32), "synth_logit": logit(np.clip(s, 1e-6, 1 - 1e-6)).astype(np.float32)})
run("sh_synth", full, te_specs=temulti(3) + te_all(), static_version="v2", params=S, extra_fn=syn_fn, ref="lgb_allte_shallow")
