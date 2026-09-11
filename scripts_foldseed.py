"""Second outer-fold split (fold seed 1) for the strongest members; averaged with the seed-0 bags later."""
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
        specs += [TESpec((c,), m=m) for c in OTHER + DIG] + [TESpec((INC,), m=m, binwidth=1000)]
    return specs
TE = temulti(3) + te_all()
S_LGB = {"max_depth": 5, "num_leaves": 32, "min_child_samples": 10, "feature_fraction": 0.3, "bagging_fraction": 0.8,
         "lambda_l1": 0.071, "lambda_l2": 2.0, "max_bin": 1024}
S_CB = {"depth": 5, "rsm": 0.3, "l2_leaf_reg": 2.0, "learning_rate": 0.08, "border_count": 1024}
fs = int(sys.argv[1])
for s in (0, 1):
    run(f"cbH_k20_fs{fs}_s{s}", full, te_specs=TE, model="cb", static_version="v2", n_folds=20, fold_seed=fs, seed_te=10 + s,
        params={**S_CB, "random_seed": 10 + s}, ref="cbF_k20_s0")
    run(f"lgbH_k20_fs{fs}_s{s}", full, te_specs=TE, static_version="v2", n_folds=20, fold_seed=fs, seed_te=10 + s,
        params={**S_LGB, "seed": 10 + s}, ref="lgbF_k20_s0")
