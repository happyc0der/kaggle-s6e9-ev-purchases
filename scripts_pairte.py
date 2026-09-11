"""Pairwise TEs among low-cardinality columns under the shallow recipe (depth-5 trees can't build these interactions)."""
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
PAIRS = [("Environmental_Concern_Level", "Subsidy_Available"), ("Range_Anxiety_Level", "Subsidy_Available"),
         ("Environmental_Concern_Level", "Range_Anxiety_Level"), ("City_Type", "Home_Charging_Possible"),
         ("Charging_Stations_Near_Home", "Home_Charging_Possible"), ("Charging_Stations_Near_Work", "City_Type"),
         ("Age", "Subsidy_Available"), ("Age", "Environmental_Concern_Level"), ("Charging_Stations_Near_Home", "Charging_Stations_Near_Work")]
TRIPLES = [("Environmental_Concern_Level", "Subsidy_Available", "Range_Anxiety_Level"),
           ("Environmental_Concern_Level", "Subsidy_Available", "Home_Charging_Possible")]
REF = "lgb_allte_shallow"
run("sh_pairte", full, te_specs=TE + [TESpec(p, m=10) for p in PAIRS], static_version="v2", params=S, ref=REF)
run("sh_pairte_tri", full, te_specs=TE + [TESpec(p, m=10) for p in PAIRS + TRIPLES] + [TESpec((INC, "Subsidy_Available"), m=10, binwidth=1000),
    TESpec((INC, "Environmental_Concern_Level"), m=10, binwidth=1000)], static_version="v2", params=S, ref="sh_pairte")
