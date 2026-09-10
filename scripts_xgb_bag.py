import json, numpy as np
from sklearn.metrics import roc_auc_score
from src.cv import run, get_data, load_oof, EXP
from src.features.static import GROUPS
from src.features.target_enc import TESpec
INC, COM = "Annual_Income_USD", "Daily_Commute_km"
full = GROUPS["raw"] + GROUPS["inc_stats"] + GROUPS["com_stats"] + GROUPS["digits"]
def temulti(m=3):
    return [TESpec((INC,), m=m), TESpec((COM,), m=m), TESpec((INC,), m=m, binwidth=100), TESpec((INC,), m=m, binwidth=500),
            TESpec((INC,), m=m, binwidth=2000), TESpec((COM,), m=m, binwidth=1), TESpec((COM,), m=m, binwidth=5)]
names = []
for s in (0, 1, 2):
    run(f"xgb_m3_teseed{s}", full, te_specs=temulti(3), model="xgb", static_version="v2", seed_te=s,
        params={"max_bin": 1024, "seed": s}, ref="xgb_full")
    names.append(f"xgb_m3_teseed{s}")
_, _, _, _, y, _ = get_data("v2")
o = np.mean([load_oof(n)[0] for n in names], 0); t = np.mean([load_oof(n)[1] for n in names], 0)
auc = roc_auc_score(y, o); print("xgb TE-seed bag of 3:", round(auc, 6))
d = EXP / "xgb_m3_tebag3"; d.mkdir(exist_ok=True); np.save(d / "oof.npy", o); np.save(d / "test.npy", t)
json.dump(dict(name="xgb_m3_tebag3", sig="avg", auc=auc, folds_done=list(range(10)), cols=[], model="xgb"), open(d / "meta.json", "w"))
with open(EXP / "results.csv", "a") as f:
    f.write(f"bag,xgb_m3_tebag3,xgb,{auc:.6f},0,10,xgb_full,,\n")
