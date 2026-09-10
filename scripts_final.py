"""Assemble a submission from experiment members: OOF-optimised weights, hard-edge post-processing, CSV."""
import sys, json, numpy as np
from sklearn.metrics import roc_auc_score
from src.blend import collect, optimize_weights
from src.cv import get_data
from src.postprocess import hard_edges
from src.submit import write_submission

names = sys.argv[2:] if len(sys.argv) > 2 else ["lgb_full_temulti_m3", "xgb_full", "nn_E9_avg3"]
tag = sys.argv[1] if len(sys.argv) > 1 else "blend"
tr, te, _, _, y, _ = get_data("v2")
O, T, y = collect(names)
w, auc = optimize_weights(O, y)
oof = O @ w
oof_pp = hard_edges(oof, tr)
print("members:", dict(zip(names, np.round(w, 3))))
print(f"blend OOF {auc:.6f} | with hard edges {roc_auc_score(y, oof_pp):.6f}")
pred = hard_edges(T @ w, te)
path = write_submission(pred, f"{tag}_{auc:.5f}", cv=auc)
json.dump(dict(names=names, weights=w.tolist(), oof_auc=auc), open(path.replace(".csv", ".json"), "w"))
print("wrote", path)
