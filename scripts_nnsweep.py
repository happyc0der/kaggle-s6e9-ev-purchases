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
cfgs = {
  "A_reg":   dict(lr=1e-3, wd=1e-4, drop=0.3, epochs=20, patience=5),
  "B_small": dict(lr=1e-3, emb_inc=8, emb_com=4, min_count=10, epochs=20, patience=5),
  "C_slow":  dict(lr=5e-4, bs=1024, epochs=20, patience=5),
  "D_narrow":dict(lr=1e-3, hidden=(256,128), drop=0.3, epochs=20, patience=5),
  "E_noemb": dict(lr=1e-3, min_count=10**9, epochs=20, patience=5),
  "F_reg2":  dict(lr=1e-3, wd=1e-3, drop=0.4, emb_inc=8, emb_com=4, min_count=5, epochs=25, patience=6),
}
out = {}
for k, p in cfgs.items():
    if k[0] not in "EF": continue
    m = run(f"nnsweep_{k}", full, te_specs=te_multi, model="nn", params=p, static_version="v2", folds_subset=[0, 1])
    out[k] = m["fold_auc"]
ref = json.load(open("experiments/nn_full/meta.json"))["fold_auc"]
lgb = json.load(open("experiments/lgb_full_temulti/meta.json"))["fold_auc"]
print("REF nn_full", ref["0"], ref["1"], " LGB", lgb["0"], lgb["1"])
for k, v in out.items(): print(k, v)
