#!/bin/zsh
cd /Users/happyc0der/Projects/kaggle-s6e9
for s in 1 2 3 4 5; do .venv/bin/python scripts_nn_final_h.py $s > experiments/log_nnH_s$s.txt 2>&1; done
.venv/bin/python - <<'PY' >> experiments/log_nn_k20_avg.txt 2>&1
import json, numpy as np
from sklearn.metrics import roc_auc_score
from src.cv import get_data, load_oof, EXP
_, _, _, _, y, _ = get_data("v2")
names = [f"nnH_k20_s{s}" for s in range(1, 6)] + ["nnG_k20_s0"]
o = np.mean([load_oof(n)[0] for n in names], 0); t = np.mean([load_oof(n)[1] for n in names], 0)
auc = roc_auc_score(y, o)
d = EXP / "nnH_k20_avg6"; d.mkdir(exist_ok=True); np.save(d / "oof.npy", o); np.save(d / "test.npy", t)
json.dump(dict(name="nnH_k20_avg6", sig="avg", auc=auc, folds_done=list(range(20)), cols=[], model="nn"), open(d / "meta.json", "w"))
open(EXP / "results.csv", "a").write(f"bag,nnH_k20_avg6,nn,{auc:.6f},0,20,nnG_k20_avg10,,\n")
PY
