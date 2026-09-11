"""Raw quantized income/commute columns (target-free) as in the public single-LGBM notebook."""
import numpy as np, pandas as pd
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
        specs += [TESpec((c,), m=m) for c in OTHER + DIG] + [TESpec((INC,), m=m, binwidth=1000)]
    return specs
S = {"max_depth": 5, "num_leaves": 32, "min_child_samples": 10, "feature_fraction": 0.3, "bagging_fraction": 0.8,
     "lambda_l1": 0.071, "lambda_l2": 2.0, "max_bin": 1024}
def quant(tr, te, static):
    inc = np.round(static[INC].to_numpy(float)).astype(np.int64); km = np.round(static[COM].to_numpy(float) * 10).astype(np.int64)
    out = {f"inc_q{d}": inc // d for d in (50, 100, 250, 500, 1000, 2500, 5000)}
    out.update({f"km_q{d}": km // d for d in (5, 10, 25, 50)})
    out["inc_res100"] = inc % 100; out["inc_res1000"] = inc % 1000; out["km_res100"] = km % 100
    return pd.DataFrame({k: np.asarray(v, dtype=np.int32) for k, v in out.items()})
run("sh_quant", full, te_specs=temulti(3) + te_all(), static_version="v2", params=S, extra_fn=quant, ref="lgb_allte_shallow")
