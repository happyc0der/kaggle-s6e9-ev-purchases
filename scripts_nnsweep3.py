import json, numpy as np, pandas as pd
from src.cv import run
from src.features.static import GROUPS
from src.features.target_enc import TESpec
INC, COM = "Annual_Income_USD", "Daily_Commute_km"
base = GROUPS["raw"] + GROUPS["inc_stats"] + GROUPS["com_stats"]
te1 = [TESpec((INC,), m=10), TESpec((COM,), m=10)]
te_multi = te1 + [TESpec((INC,), m=10, binwidth=100), TESpec((INC,), m=10, binwidth=500),
                  TESpec((INC,), m=10, binwidth=2000), TESpec((COM,), m=10, binwidth=1), TESpec((COM,), m=10, binwidth=5)]
full = base + GROUPS["digits"]
ic = ("Environmental_Concern_Level", "Number_of_Cars_Owned", "Charging_Stations_Near_Home", "Charging_Stations_Near_Work", "Age")
E8 = dict(lr=1e-3, min_count=10**9, extra_cats=ic, emb_cat=6, wd=1e-4, drop=0.3, epochs=25, patience=6)

def bins_fn(tr, te, static):
    inc = static[INC].to_numpy(float); com = static[COM].to_numpy(float)
    return pd.DataFrame({"inc_bin1k": np.floor(inc / 1000).astype(np.int64), "com_bin1": np.floor(com).astype(np.int64),
                         "inc_bin250": np.floor(inc / 250).astype(np.int64)})

cfgs = {
  "E9_morereg": ({**E8, "drop": 0.4, "wd": 3e-4}, None, ()),
  "E10_long":   ({**E8, "emb_cat": 8, "epochs": 35, "patience": 8}, None, ()),
  "E11_bins":   ({**E8, "extra_cats": ic + ("inc_bin1k", "com_bin1")}, bins_fn, ()),
  "E12_bins250":({**E8, "extra_cats": ic + ("inc_bin250", "com_bin1")}, bins_fn, ()),
  "E13_freqemb":({**E8, "min_count": 50, "emb_inc": 6, "emb_com": 4}, None, ()),
}
out = {}
for k, (p, fn, _) in cfgs.items():
    m = run(f"nnsweep_{k}", full, te_specs=te_multi, model="nn", params=p, static_version="v2", folds_subset=[0, 1], extra_fn=fn)
    out[k] = m["fold_auc"]
print("REF E8", json.load(open("experiments/nnsweep_E8_cats_reg/meta.json"))["fold_auc"])
for k, v in out.items(): print(k, v)
