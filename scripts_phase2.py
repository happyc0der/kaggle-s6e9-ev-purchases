"""Phase 2: diverse members on the best feature set (NN embeddings, CatBoost native categoricals, XGB) + blend."""
import sys
from src.cv import run
from src.features.static import GROUPS
from src.features.target_enc import TESpec

INC, COM = "Annual_Income_USD", "Daily_Commute_km"
raw = GROUPS["raw"]
base = raw + GROUPS["inc_stats"] + GROUPS["com_stats"]
full = base + GROUPS["origmatch"] + GROUPS["digits"]
te1 = [TESpec((INC,), m=10), TESpec((COM,), m=10)]
te_multi = te1 + [TESpec((INC,), m=10, binwidth=100), TESpec((INC,), m=10, binwidth=500),
                  TESpec((INC,), m=10, binwidth=2000), TESpec((COM,), m=10, binwidth=1), TESpec((COM,), m=10, binwidth=5)]
stage = sys.argv[1] if len(sys.argv) > 1 else "all"

if stage in ("nn", "all"):
    run("nn_full", full, te_specs=te_multi, model="nn", static_version="v2", ref="lgb_full_te1")
if stage in ("cb", "all"):
    # income/commute as native categoricals -> CatBoost builds its own ordered target stats + combinations
    run("cb_full_cat", full, te_specs=te1, model="cb", static_version="v2", cat_cols=[INC, COM],
        params={"learning_rate": 0.06, "depth": 8}, ref="lgb_full_te1")
if stage in ("xgb", "all"):
    run("xgb_full", full, te_specs=te_multi, model="xgb", static_version="v2", params={"max_bin": 1024}, ref="lgb_full_te1")
