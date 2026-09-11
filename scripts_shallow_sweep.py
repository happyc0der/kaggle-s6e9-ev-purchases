"""Sweep around the shallow / column-subsampled LightGBM with all-column TEs."""
from src.cv import run
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
S = {"max_depth": 5, "num_leaves": 32, "min_child_samples": 10, "feature_fraction": 0.3, "bagging_fraction": 0.8,
     "lambda_l1": 0.071, "lambda_l2": 2.0, "max_bin": 1024}
TE = temulti(3) + te_all()
REF = "lgb_allte_shallow"
run("sh_ff02", full, te_specs=TE, static_version="v2", params={**S, "feature_fraction": 0.2}, ref=REF)
run("sh_ff04", full, te_specs=TE, static_version="v2", params={**S, "feature_fraction": 0.4}, ref=REF)
run("sh_d4", full, te_specs=TE, static_version="v2", params={**S, "max_depth": 4, "num_leaves": 16}, ref=REF)
run("sh_d6", full, te_specs=TE, static_version="v2", params={**S, "max_depth": 6, "num_leaves": 48}, ref=REF)
run("sh_mc30", full, te_specs=TE, static_version="v2", params={**S, "min_child_samples": 30}, ref=REF)
run("sh_lr01", full, te_specs=TE, static_version="v2", params={**S, "learning_rate": 0.01}, ref=REF)
run("sh_te3", full, te_specs=temulti(3) + te_all((3, 10, 100)), static_version="v2", params=S, ref=REF)
run("sh_extras", full + GROUPS["extras"], te_specs=TE, static_version="v2", params=S, ref=REF)
