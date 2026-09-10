"""Phase 0/1: baseline, noise control, and the public-frontier feature families with paired ablations."""
import sys
from src.cv import run
from src.features.static import GROUPS
from src.features.target_enc import TESpec

INC, COM = "Annual_Income_USD", "Daily_Commute_km"
raw = GROUPS["raw"]
stage = sys.argv[1] if len(sys.argv) > 1 else "all"

if stage in ("base", "all"):
    run("lgb_raw", raw)
    run("lgb_raw_noise", raw, noise=True, ref="lgb_raw")

if stage in ("fe", "all"):
    run("lgb_stats", raw + GROUPS["inc_stats"] + GROUPS["com_stats"], ref="lgb_raw")
    te1 = [TESpec((INC,), m=10), TESpec((COM,), m=10)]
    run("lgb_te", raw, te_specs=te1, ref="lgb_raw")
    run("lgb_stats_te", raw + GROUPS["inc_stats"] + GROUPS["com_stats"], te_specs=te1, ref="lgb_stats")
    base = raw + GROUPS["inc_stats"] + GROUPS["com_stats"]
    run("lgb_stats_te_digits", base + GROUPS["digits"], te_specs=te1, ref="lgb_stats_te")
    run("lgb_stats_te_dens", base + GROUPS["density"], te_specs=te1, ref="lgb_stats_te")
    run("lgb_stats_te_pairs", base + GROUPS["pairs"], te_specs=te1, ref="lgb_stats_te")
    te_multi = te1 + [TESpec((INC,), m=10, binwidth=100), TESpec((INC,), m=10, binwidth=500),
                      TESpec((INC,), m=10, binwidth=2000), TESpec((COM,), m=10, binwidth=1), TESpec((COM,), m=10, binwidth=5)]
    run("lgb_stats_temulti", base, te_specs=te_multi, ref="lgb_stats_te")
