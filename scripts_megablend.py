"""Blend our members with every public member that ships an out-of-fold file. Weights fit on OOF (rank space), nested check."""
import glob, json, numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from src.cv import get_data, load_oof
from src.postprocess import hard_edges
from src.submit import write_submission
tr, te, orig, static, y, folds = get_data("v2")
assert np.array_equal(tr.id.to_numpy(), np.arange(len(tr))) and np.array_equal(te.id.to_numpy(), 668665 + np.arange(len(te)))
P = "data/public"
rk = lambda v: rankdata(v) / len(v)
O, T, names = [], [], []
def add(name, oof, test):
    oof, test = np.asarray(oof, float), np.asarray(test, float)
    assert len(oof) == len(tr) and len(test) == len(te) and np.isfinite(oof).all() and np.isfinite(test).all(), name
    a = roc_auc_score(y, oof)
    if a < 0.9: print(f"  skip {name}: OOF AUC {a:.4f} (misaligned or weak)"); return
    O.append(rk(oof)); T.append(rk(test)); names.append(name); print(f"  {name:36s} OOF {a:.6f}")
def by_id(df, col): return df.set_index("id")[col]
# ours
for n in ["lgb_final_bag10", "cb_final_bag8", "nn_final_bag15"]:
    o, t = load_oof(n); add("ours_" + n, o, t)
# megayak six views + realmlp
so, st = pd.read_csv(f"{P}/sixviews/oof_six_views.csv"), pd.read_csv(f"{P}/sixviews/test_six_views.csv")
for c in ["A_lgbm_triple_te_digits_3seed", "B_xgb_on_A_features", "C_no_digits_windows_lift_sm2_30_300", "D_no_exact_key_ladder_windows",
          "E_ladder25_250_2500_lift_sm5_50_500", "F_exact_rate_as_init_score"]:
    add("six_" + c[:24], by_id(so, c).reindex(tr.id).to_numpy(), by_id(st, c).reindex(te.id).to_numpy())
ro, rt = pd.read_csv(f"{P}/sixviews/oof_realmlp_g.csv"), pd.read_csv(f"{P}/sixviews/test_realmlp_g.csv")
add("realmlp_G_3seed", by_id(ro, "G_realmlp_3seed").reindex(tr.id).to_numpy(), by_id(rt, "G_realmlp_3seed").reindex(te.id).to_numpy())
# naji xgb
add("naji_xgb", by_id(pd.read_csv(f"{P}/naji_xgb/oof_XGBOOST.csv"), "OOF_Pred").reindex(tr.id).to_numpy(),
    by_id(pd.read_csv(f"{P}/naji_xgb/test_XGBOOST.csv"), "Will_Buy_EV").reindex(te.id).to_numpy())
# yekenot realmlp
add("yekenot_realmlp", by_id(pd.read_csv(f"{P}/realmlp/oof_preds.csv"), "Will_Buy_EV").reindex(tr.id).to_numpy(),
    by_id(pd.read_csv(f"{P}/realmlp/submission.csv"), "Will_Buy_EV").reindex(te.id).to_numpy())
# hirge library: every oof_X.npy with a matching test_X.npy (assumed train/test row order)
for f in sorted(glob.glob(f"{P}/hirge/oof_*.npy")):
    n = f.split("oof_")[-1][:-4]; tf = f"{P}/hirge/test_{n}.npy"
    if not glob.glob(tf): continue
    o, t = np.load(f), np.load(tf)
    if o.ndim != 1 or t.ndim != 1 or len(o) != len(tr) or len(t) != len(te): continue
    add("hirge_" + n, o, t)
O, T = np.column_stack(O), np.column_stack(T)
print("\nmembers:", len(names))
pos = y == 1
def auc_of(score, mask):
    s = score[mask]; yy = pos[mask]; r = rankdata(s); n1 = yy.sum(); n0 = len(yy) - n1
    return (r[yy].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
def hill_climb(mask, steps=40):
    k = O.shape[1]; counts = np.zeros(k, int)
    i0 = max(range(k), key=lambda i: auc_of(O[:, i], mask)); counts[i0] = 1; cur = O[:, i0].copy(); cur_auc = auc_of(cur, mask)
    for _ in range(steps - 1):
        tot = counts.sum()
        a, i = max((auc_of((cur * tot + O[:, i]) / (tot + 1), mask), i) for i in range(k))
        if a <= cur_auc + 1e-9: break
        counts[i] += 1; cur = (cur * tot + O[:, i]) / (tot + 1); cur_auc = a
    return counts / counts.sum()
gains = []
for f in range(10):
    trm, vam = folds != f, folds == f
    w = hill_climb(trm); gains.append(auc_of(O @ w, vam) - max(auc_of(O[:, i], vam) for i in range(O.shape[1])))
print(f"nested gain over best single member: mean {np.mean(gains)*1e5:+.2f}e-5, positive in {(np.array(gains) > 0).sum()}/10 folds")
W = hill_climb(np.ones(len(y), bool))
print("weights:", {n: round(w, 3) for n, w in zip(names, W) if w > 0})
oof_blend = O @ W; test_blend = T @ W
print(f"blend OOF {roc_auc_score(y, oof_blend):.6f} | with hard edges {roc_auc_score(y, hard_edges(oof_blend, tr)):.6f}")
np.save("experiments/megablend_oof.npy", oof_blend); np.save("experiments/megablend_test.npy", test_blend)
json.dump(dict(names=names, weights=W.tolist()), open("experiments/megablend_weights.json", "w"))
pred = hard_edges(test_blend, te)
path = write_submission(pred, f"mega1_{roc_auc_score(y, hard_edges(oof_blend, tr)):.5f}", cv=float(roc_auc_score(y, hard_edges(oof_blend, tr))))
print("wrote", path)
