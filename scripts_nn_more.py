"""More NN seeds + feature-view variants for blend diversity."""
import json, numpy as np
from sklearn.metrics import roc_auc_score
from src.cv import run, get_data, load_oof, EXP
from src.features.static import GROUPS
from src.features.target_enc import TESpec
INC, COM = "Annual_Income_USD", "Daily_Commute_km"
base = GROUPS["raw"] + GROUPS["inc_stats"] + GROUPS["com_stats"]
full = base + GROUPS["digits"]
te1 = [TESpec((INC,), m=10), TESpec((COM,), m=10)]
te_multi = te1 + [TESpec((INC,), m=10, binwidth=100), TESpec((INC,), m=10, binwidth=500),
                  TESpec((INC,), m=10, binwidth=2000), TESpec((COM,), m=10, binwidth=1), TESpec((COM,), m=10, binwidth=5)]
ic = ("Environmental_Concern_Level", "Number_of_Cars_Owned", "Charging_Stations_Near_Home", "Charging_Stations_Near_Work", "Age")
E9 = dict(lr=1e-3, min_count=10**9, extra_cats=ic, emb_cat=6, wd=3e-4, drop=0.4, epochs=25, patience=6)
for s in (3, 4):
    run(f"nn_E9_s{s}", full, te_specs=te_multi, model="nn", params={**E9, "seed": s}, static_version="v2")
_, _, _, _, y, _ = get_data("v2")
names = [f"nn_E9_s{s}" for s in range(5)]
o = np.mean([load_oof(n)[0] for n in names], 0); t = np.mean([load_oof(n)[1] for n in names], 0)
auc = roc_auc_score(y, o); print("nn 5-seed avg", round(auc, 6))
d = EXP / "nn_E9_avg5"; d.mkdir(exist_ok=True); np.save(d / "oof.npy", o); np.save(d / "test.npy", t)
json.dump(dict(name="nn_E9_avg5", sig="avg", auc=auc, folds_done=list(range(10)), cols=[], model="nn"), open(d / "meta.json", "w"))
with open(EXP / "results.csv", "a") as f:
    f.write(f"bag,nn_E9_avg5,nn,{auc:.6f},0,10,nn_E9_avg3,,\n")
# feature-view variants
run("nn_E9_te1only", base, te_specs=te1, model="nn", params=E9, static_version="v2", ref="nn_E9_s0")
run("nn_E9_rawte", GROUPS["raw"], te_specs=te_multi, model="nn", params=E9, static_version="v2", ref="nn_E9_s0")
