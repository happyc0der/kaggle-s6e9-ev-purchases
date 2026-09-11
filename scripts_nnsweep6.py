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
ic = ("Environmental_Concern_Level", "Number_of_Cars_Owned", "Charging_Stations_Near_Home", "Charging_Stations_Near_Work", "Age")
E9 = dict(lr=1e-3, min_count=10**9, extra_cats=ic, emb_cat=6, wd=3e-4, drop=0.4, epochs=25, patience=6)
cfgs = {"EMA999": {**E9, "ema": 0.999}, "EMA9995_long": {**E9, "ema": 0.9995, "epochs": 35, "patience": 8},
        "EMA999_lr2": {**E9, "ema": 0.999, "lr": 2e-3}}
out = {}
for k, p in cfgs.items():
    out[k] = run(f"nnsweep_{k}", full, te_specs=te_multi, model="nn", params=p, static_version="v2", folds_subset=[0, 1])["fold_auc"]
print("REF E9", json.load(open("experiments/nnsweep_E9_morereg/meta.json"))["fold_auc"])
for k, v in out.items(): print(k, v)
