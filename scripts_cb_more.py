"""CatBoost: extra 20-fold TE seeds for a bag, and a small tuning sweep under the shallow all-TE recipe."""
import sys, json, numpy as np
from sklearn.metrics import roc_auc_score
from src.cv import run, get_data, load_oof, EXP
from src.features.static import GROUPS
from src.features.target_enc import TESpec
INC, COM = "Annual_Income_USD", "Daily_Commute_km"
full = GROUPS["raw"] + GROUPS["inc_stats"] + GROUPS["com_stats"] + GROUPS["digits"]
def temulti(m=3):
    return [TESpec((INC,), m=m), TESpec((COM,), m=m), TESpec((INC,), m=m, binwidth=100), TESpec((INC,), m=m, binwidth=500),
            TESpec((INC,), m=m, binwidth=2000), TESpec((COM,), m=m, binwidth=1), TESpec((COM,), m=m, binwidth=5)]
OTHER = ["Age", "Charging_Stations_Near_Home", "Charging_Stations_Near_Work", "Environmental_Concern_Level", "Number_of_Cars_Owned",
         "Gender", "City_Type", "Current_Car_Type", "Home_Charging_Possible", "Subsidy_Available", "Range_Anxiety_Level"]
DIG = ["inc_d1", "inc_d2", "inc_d3", "inc_d4", "com_d_dec", "com_d1"]
def te_all(ms=(10, 100)):
    specs = []
    for m in ms:
        specs += [TESpec((c,), m=m) for c in OTHER + DIG]
        specs += [TESpec((INC,), m=m, binwidth=1000)]
    return specs
TE = temulti(3) + te_all()
S_CB = {"depth": 5, "rsm": 0.3, "l2_leaf_reg": 2.0, "learning_rate": 0.08, "border_count": 1024}
stage = sys.argv[1]
if stage == "sweep":
    run("cbsw_d6", full, te_specs=TE, model="cb", static_version="v2", params={**S_CB, "depth": 6}, ref="cb_sh_allte")
    run("cbsw_rsm02", full, te_specs=TE, model="cb", static_version="v2", params={**S_CB, "rsm": 0.2}, ref="cb_sh_allte")
    run("cbsw_lr05", full, te_specs=TE, model="cb", static_version="v2", params={**S_CB, "learning_rate": 0.05}, ref="cb_sh_allte")
    run("cbsw_l2_5", full, te_specs=TE, model="cb", static_version="v2", params={**S_CB, "l2_leaf_reg": 5.0}, ref="cb_sh_allte")
elif stage == "bag":
    names = ["cbF_k20_s0"]
    for s in (1, 2):
        run(f"cbF_k20_s{s}", full, te_specs=TE, model="cb", static_version="v2", n_folds=20, seed_te=s, params={**S_CB, "random_seed": s}, ref="cbF_k20_s0")
        names.append(f"cbF_k20_s{s}")
    _, _, _, _, y, _ = get_data("v2")
    o = np.mean([load_oof(n)[0] for n in names], 0); t = np.mean([load_oof(n)[1] for n in names], 0)
    auc = roc_auc_score(y, o); d = EXP / "cbF_k20_bag3"; d.mkdir(exist_ok=True); np.save(d / "oof.npy", o); np.save(d / "test.npy", t)
    json.dump(dict(name="cbF_k20_bag3", sig="avg", auc=auc, folds_done=list(range(20)), cols=[], model="cb"), open(d / "meta.json", "w"))
    open(EXP / "results.csv", "a").write(f"bag,cbF_k20_bag3,cb,{auc:.6f},0,20,,,\n"); print("cbF_k20_bag3", round(auc, 6))
