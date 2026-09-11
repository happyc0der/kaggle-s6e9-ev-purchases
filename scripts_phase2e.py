"""Phase 2e: more TE-seed variants for the 20-fold bags + 20-fold feature-fraction member."""
from src.cv import run
from src.features.static import GROUPS
from src.features.target_enc import TESpec
INC, COM = "Annual_Income_USD", "Daily_Commute_km"
full = GROUPS["raw"] + GROUPS["inc_stats"] + GROUPS["com_stats"] + GROUPS["digits"]
def temulti(m=3):
    return [TESpec((INC,), m=m), TESpec((COM,), m=m), TESpec((INC,), m=m, binwidth=100), TESpec((INC,), m=m, binwidth=500),
            TESpec((INC,), m=m, binwidth=2000), TESpec((COM,), m=m, binwidth=1), TESpec((COM,), m=m, binwidth=5)]
for s in (1, 2):
    run(f"cb_m3_k20_teseed{s}", full, te_specs=temulti(3), model="cb", static_version="v2", n_folds=20, seed_te=s,
        params={"learning_rate": 0.08, "depth": 8, "random_seed": s}, ref="cb_m3_k20")
run("lgb_m3_ff04_k20", full, te_specs=temulti(3), static_version="v2", n_folds=20,
    params={"feature_fraction": 0.4, "num_leaves": 127, "min_child_samples": 300}, ref="lgb_m3_ff04")
for s in (3, 4):
    run(f"lgb_m3_k20_teseed{s}", full, te_specs=temulti(3), static_version="v2", n_folds=20, seed_te=s, params={"seed": s}, ref="lgb_m3_k20")
    run(f"xgb_m3_k20_teseed{s}", full, te_specs=temulti(3), model="xgb", static_version="v2", n_folds=20, seed_te=s,
        params={"max_bin": 1024, "seed": s}, ref="xgb_m3_k20")
