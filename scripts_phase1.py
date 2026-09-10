"""Phase 1b: original-row similarity features and combined sets on top of the public recipe."""
from src.cv import run
from src.features.static import GROUPS
from src.features.target_enc import TESpec

INC, COM = "Annual_Income_USD", "Daily_Commute_km"
raw = GROUPS["raw"]
base = raw + GROUPS["inc_stats"] + GROUPS["com_stats"]
te1 = [TESpec((INC,), m=10), TESpec((COM,), m=10)]
te_multi = te1 + [TESpec((INC,), m=10, binwidth=100), TESpec((INC,), m=10, binwidth=500),
                  TESpec((INC,), m=10, binwidth=2000), TESpec((COM,), m=10, binwidth=1), TESpec((COM,), m=10, binwidth=5)]

run("lgb_stats_te_om", base + GROUPS["origmatch"], te_specs=te1, static_version="v2", ref="lgb_stats_te")
run("lgb_all_v2", base + GROUPS["origmatch"] + GROUPS["digits"] + GROUPS["density"] + GROUPS["pairs"],
    te_specs=te_multi, static_version="v2", ref="lgb_stats_te")
run("lgb_all_v2_bin4k", base + GROUPS["origmatch"] + GROUPS["digits"] + GROUPS["density"] + GROUPS["pairs"],
    te_specs=te_multi, static_version="v2", params={"max_bin": 4095}, ref="lgb_all_v2")
