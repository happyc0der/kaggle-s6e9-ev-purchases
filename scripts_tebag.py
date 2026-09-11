"""Bagged training-side TE (average over inner-fold seeds) and feature-group elimination, shallow recipe, k10."""
import sys
from src.cv import run
from src.features.static import GROUPS
from src.features.target_enc import TESpec
INC, COM = "Annual_Income_USD", "Daily_Commute_km"
full = GROUPS["raw"] + GROUPS["inc_stats"] + GROUPS["com_stats"] + GROUPS["digits"]
def temulti(m=3, **kw):
    return [TESpec((INC,), m=m, **kw), TESpec((COM,), m=m, **kw), TESpec((INC,), m=m, binwidth=100, **kw), TESpec((INC,), m=m, binwidth=500, **kw),
            TESpec((INC,), m=m, binwidth=2000, **kw), TESpec((COM,), m=m, binwidth=1, **kw), TESpec((COM,), m=m, binwidth=5, **kw)]
OTHER = ["Age", "Charging_Stations_Near_Home", "Charging_Stations_Near_Work", "Environmental_Concern_Level", "Number_of_Cars_Owned",
         "Gender", "City_Type", "Current_Car_Type", "Home_Charging_Possible", "Subsidy_Available", "Range_Anxiety_Level"]
DIG = ["inc_d1", "inc_d2", "inc_d3", "inc_d4", "com_d_dec", "com_d1"]
def te_all(ms=(10, 100), cols=None, **kw):
    cols = OTHER + DIG if cols is None else cols
    specs = []
    for m in ms:
        specs += [TESpec((c,), m=m, **kw) for c in cols] + [TESpec((INC,), m=m, binwidth=1000, **kw)]
    return specs
S = {"max_depth": 5, "num_leaves": 32, "min_child_samples": 10, "feature_fraction": 0.3, "bagging_fraction": 0.8,
     "lambda_l1": 0.071, "lambda_l2": 2.0, "max_bin": 1024}
REF = "lgb_allte_shallow"
stage = sys.argv[1]
if stage == "tebag":
    run("sh_tebag5", full, te_specs=temulti(3, nbag=5) + te_all(nbag=5), static_version="v2", params=S, ref=REF)
elif stage == "elim":
    run("sh_noDigTE", full, te_specs=temulti(3) + te_all(cols=OTHER), static_version="v2", params=S, ref=REF)
    run("sh_noStats", GROUPS["raw"] + GROUPS["digits"], te_specs=temulti(3) + te_all(), static_version="v2", params=S, ref=REF)
    run("sh_noDigits", GROUPS["raw"] + GROUPS["inc_stats"] + GROUPS["com_stats"], te_specs=temulti(3) + te_all(), static_version="v2", params=S, ref=REF)
