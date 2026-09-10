import json
from src.cv import run
from src.features.static import GROUPS
from src.features.target_enc import TESpec
INC, COM = "Annual_Income_USD", "Daily_Commute_km"
base = GROUPS["raw"] + GROUPS["inc_stats"] + GROUPS["com_stats"]
te1 = [TESpec((INC,), m=10), TESpec((COM,), m=10)]
te_multi = te1 + [TESpec((INC,), m=10, binwidth=100), TESpec((INC,), m=10, binwidth=500),
                  TESpec((INC,), m=10, binwidth=2000), TESpec((COM,), m=10, binwidth=1), TESpec((COM,), m=10, binwidth=5)]
full = base + GROUPS["digits"]
E = dict(lr=1e-3, min_count=10**9, epochs=20, patience=5)
ic = ("Environmental_Concern_Level", "Number_of_Cars_Owned", "Charging_Stations_Near_Home", "Charging_Stations_Near_Work", "Age")
cfgs = {
  "E1_reg":   {**E, "wd": 1e-4, "drop": 0.3, "epochs": 25, "patience": 6},
  "E2_wide":  {**E, "hidden": (1024, 512, 256), "drop": 0.3},
  "E3_slow":  {**E, "lr": 5e-4, "epochs": 30, "patience": 8},
  "E6_bigbs": {**E, "bs": 8192, "lr": 3e-3},
  "E7_cats":  {**E, "extra_cats": ic, "emb_cat": 6},
  "E8_cats_reg": {**E, "extra_cats": ic, "emb_cat": 6, "wd": 1e-4, "drop": 0.3, "epochs": 25, "patience": 6},
}
out = {}
for k, p in cfgs.items():
    m = run(f"nnsweep_{k}", full, te_specs=te_multi, model="nn", params=p, static_version="v2", folds_subset=[0, 1])
    out[k] = m["fold_auc"]
print("REF E_noemb", json.load(open("experiments/nnsweep_E_noemb/meta.json"))["fold_auc"])
for k, v in out.items(): print(k, v)
