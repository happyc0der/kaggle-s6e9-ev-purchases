"""Phase 1d: inner-fold count for TE, single interaction TEs."""
from src.cv import run
from src.features.static import GROUPS
from src.features.target_enc import TESpec
INC, COM = "Annual_Income_USD", "Daily_Commute_km"
base = GROUPS["raw"] + GROUPS["inc_stats"] + GROUPS["com_stats"]
full = base + GROUPS["digits"]
def temulti(m=3, inner=5):
    return [TESpec((INC,), m=m, inner=inner), TESpec((COM,), m=m, inner=inner),
            TESpec((INC,), m=m, binwidth=100, inner=inner), TESpec((INC,), m=m, binwidth=500, inner=inner),
            TESpec((INC,), m=m, binwidth=2000, inner=inner), TESpec((COM,), m=m, binwidth=1, inner=inner),
            TESpec((COM,), m=m, binwidth=5, inner=inner)]
REF = "lgb_full_temulti_m3"
run("lgb_m3_inner20", full, te_specs=temulti(3, 20), static_version="v2", ref=REF)
run("lgb_m3_inccom", full, te_specs=temulti(3) + [TESpec((INC, COM), m=3)], static_version="v2", ref=REF)
run("lgb_m3_incsub", full, te_specs=temulti(3) + [TESpec((INC, "Subsidy_Available"), m=3)], static_version="v2", ref=REF)
run("lgb_m3_incbin500sub", full, te_specs=temulti(3) + [TESpec((INC, "Subsidy_Available"), m=3, binwidth=500),
    TESpec((INC, "Environmental_Concern_Level"), m=3, binwidth=500), TESpec((INC, "Range_Anxiety_Level"), m=3, binwidth=500)],
    static_version="v2", ref=REF)
