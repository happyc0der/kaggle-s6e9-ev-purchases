"""Phase 1b: combined best set, one similarity column alone, max_bin, TE smoothing check."""
from src.cv import run
from src.features.static import GROUPS
from src.features.target_enc import TESpec

INC, COM = "Annual_Income_USD", "Daily_Commute_km"
raw = GROUPS["raw"]
base = raw + GROUPS["inc_stats"] + GROUPS["com_stats"]
te1 = [TESpec((INC,), m=10), TESpec((COM,), m=10)]
te_multi = te1 + [TESpec((INC,), m=10, binwidth=100), TESpec((INC,), m=10, binwidth=500),
                  TESpec((INC,), m=10, binwidth=2000), TESpec((COM,), m=10, binwidth=1), TESpec((COM,), m=10, binwidth=5)]
full = base + GROUPS["digits"]

run("lgb_full_temulti", full, te_specs=te_multi, static_version="v2", ref="lgb_stats_temulti")
run("lgb_full_temulti_ompair", full + ["om_pair_bestmatch"], te_specs=te_multi, static_version="v2", ref="lgb_full_temulti")
run("lgb_full_temulti_bin4k", full, te_specs=te_multi, static_version="v2", params={"max_bin": 4095}, ref="lgb_full_temulti")
te_multi_m3 = [TESpec(s.cols, m=3, binwidth=s.binwidth) for s in te_multi]
run("lgb_full_temulti_m3", full, te_specs=te_multi_m3, static_version="v2", ref="lgb_full_temulti")
