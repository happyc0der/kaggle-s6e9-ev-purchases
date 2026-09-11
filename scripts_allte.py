"""All-column target encodings (public 'Naji' recipe) inside our harness, 10 folds, paired vs our best."""
import sys
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
SHALLOW = {"max_depth": 5, "num_leaves": 32, "min_child_samples": 10, "feature_fraction": 0.3, "bagging_fraction": 0.8,
           "lambda_l1": 0.071, "lambda_l2": 2.0, "max_bin": 1024}
stage = sys.argv[1] if len(sys.argv) > 1 else "all"
REF = "lgb_full_temulti_m3"
if stage in ("feat", "all"):
    run("lgb_allte", full, te_specs=temulti(3) + te_all(), static_version="v2", ref=REF)
if stage in ("params", "all"):
    run("lgb_allte_shallow", full, te_specs=temulti(3) + te_all(), static_version="v2", params=SHALLOW, ref="lgb_allte")
if stage in ("shallow_only", "all"):
    run("lgb_m3_shallow", full, te_specs=temulti(3), static_version="v2", params=SHALLOW, ref=REF)
