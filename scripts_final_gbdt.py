"""Final GBDT members: shallow all-TE recipe for xgb/cb (k10 checks), then k20 TE-seed bags."""
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
S_LGB = {"max_depth": 5, "num_leaves": 32, "min_child_samples": 10, "feature_fraction": 0.3, "bagging_fraction": 0.8,
         "lambda_l1": 0.071, "lambda_l2": 2.0, "max_bin": 1024}
S_XGB = {"max_depth": 5, "min_child_weight": 10, "colsample_bytree": 0.3, "subsample": 0.8, "reg_alpha": 0.071, "reg_lambda": 2.0, "max_bin": 1024}
S_CB = {"depth": 5, "rsm": 0.3, "l2_leaf_reg": 2.0, "learning_rate": 0.08, "border_count": 1024}
stage = sys.argv[1]
def bag(out, names, k):
    _, _, _, _, y, _ = get_data("v2")
    o = np.mean([load_oof(n)[0] for n in names], 0); t = np.mean([load_oof(n)[1] for n in names], 0)
    auc = roc_auc_score(y, o); d = EXP / out; d.mkdir(exist_ok=True); np.save(d / "oof.npy", o); np.save(d / "test.npy", t)
    json.dump(dict(name=out, sig="avg", auc=auc, folds_done=list(range(k)), cols=[], model=out[:3]), open(d / "meta.json", "w"))
    open(EXP / "results.csv", "a").write(f"bag,{out},{out[:3]},{auc:.6f},0,{k},,,\n"); print(out, round(auc, 6))
if stage == "check":
    run("xgb_sh_allte", full, te_specs=TE, model="xgb", static_version="v2", params=S_XGB, ref="xgb_m3_teseed0")
    run("cb_sh_allte", full, te_specs=TE, model="cb", static_version="v2", params=S_CB, ref="cb_m3_plain")
elif stage == "lgb20":
    names = []
    for s in (0, 1, 2):
        run(f"lgbF_k20_s{s}", full, te_specs=TE, static_version="v2", n_folds=20, seed_te=s, params={**S_LGB, "seed": s}, ref="lgb_allte_shallow")
        names.append(f"lgbF_k20_s{s}")
    bag("lgbF_k20_bag3", names, 20)
elif stage == "xgb20":
    names = []
    for s in (0, 1, 2):
        run(f"xgbF_k20_s{s}", full, te_specs=TE, model="xgb", static_version="v2", n_folds=20, seed_te=s, params={**S_XGB, "seed": s}, ref="xgb_sh_allte")
        names.append(f"xgbF_k20_s{s}")
    bag("xgbF_k20_bag3", names, 20)
elif stage == "cb20":
    run("cbF_k20_s0", full, te_specs=TE, model="cb", static_version="v2", n_folds=20, params=S_CB, ref="cb_sh_allte")
