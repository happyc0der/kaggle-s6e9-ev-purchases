import sys
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
CFG = dict(lr=2e-3, ema=0.999, min_count=10**9, extra_cats=ic, emb_cat=6, wd=3e-4, drop=0.4, epochs=25, patience=6)
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
run(f"nn_EMA_k20_s{seed}", full, te_specs=te_multi, model="nn", params={**CFG, "seed": seed}, static_version="v2", n_folds=20, ref="nn_E9_k20_s0")
