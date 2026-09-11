#!/bin/zsh
cd /Users/happyc0der/Projects/kaggle-s6e9
for s in 5 6 7 8 9; do .venv/bin/python scripts_nn_final.py $s > experiments/log_nnG_s$s.txt 2>&1; done
.venv/bin/python - <<'PY' >> experiments/log_nn_k20_avg.txt 2>&1
import json, numpy as np
from sklearn.metrics import roc_auc_score
from src.cv import get_data, load_oof, EXP
_, _, _, _, y, _ = get_data("v2")
names = [f"nnG_k20_s{s}" for s in range(10)]
o = np.mean([load_oof(n)[0] for n in names], 0); t = np.mean([load_oof(n)[1] for n in names], 0)
auc = roc_auc_score(y, o)
d = EXP / "nnG_k20_avg10"; d.mkdir(exist_ok=True); np.save(d / "oof.npy", o); np.save(d / "test.npy", t)
json.dump(dict(name="nnG_k20_avg10", sig="avg", auc=auc, folds_done=list(range(20)), cols=[], model="nn"), open(d / "meta.json", "w"))
open(EXP / "results.csv", "a").write(f"bag,nnG_k20_avg10,nn,{auc:.6f},0,20,nnG_k20_avg5,,\n")
PY
