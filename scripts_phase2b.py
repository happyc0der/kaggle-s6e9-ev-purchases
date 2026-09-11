"""Phase 2b: structurally different GBDT members for blend diversity (same best feature set)."""
from src.cv import run
from src.features.static import GROUPS
from src.features.target_enc import TESpec
INC, COM = "Annual_Income_USD", "Daily_Commute_km"
full = GROUPS["raw"] + GROUPS["inc_stats"] + GROUPS["com_stats"] + GROUPS["digits"]
def temulti(m=3):
    return [TESpec((INC,), m=m), TESpec((COM,), m=m), TESpec((INC,), m=m, binwidth=100), TESpec((INC,), m=m, binwidth=500),
            TESpec((INC,), m=m, binwidth=2000), TESpec((COM,), m=m, binwidth=1), TESpec((COM,), m=m, binwidth=5)]
REF = "lgb_full_temulti_m3"
run("lgb_m3_extratrees", full, te_specs=temulti(3), static_version="v2", params={"extra_trees": True, "learning_rate": 0.03}, ref=REF)
run("lgb_m3_ff04", full, te_specs=temulti(3), static_version="v2", params={"feature_fraction": 0.4, "num_leaves": 127, "min_child_samples": 300}, ref=REF)
run("cb_m3_plain", full, te_specs=temulti(3), model="cb", static_version="v2", params={"learning_rate": 0.08, "depth": 8}, ref=REF)
