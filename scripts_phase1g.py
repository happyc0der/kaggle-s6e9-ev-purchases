"""Phase 1g: residual target encodings (label minus formula probability) alongside the plain ones."""
from src.cv import run
from src.features.static import GROUPS
from src.features.target_enc import TESpec
INC, COM = "Annual_Income_USD", "Daily_Commute_km"
full = GROUPS["raw"] + GROUPS["inc_stats"] + GROUPS["com_stats"] + GROUPS["digits"]
def temulti(m=3, **kw):
    return [TESpec((INC,), m=m, **kw), TESpec((COM,), m=m, **kw), TESpec((INC,), m=m, binwidth=100, **kw),
            TESpec((INC,), m=m, binwidth=500, **kw), TESpec((INC,), m=m, binwidth=2000, **kw),
            TESpec((COM,), m=m, binwidth=1, **kw), TESpec((COM,), m=m, binwidth=5, **kw)]
REF = "lgb_full_temulti_m3"
run("lgb_m3_resid_both", full, te_specs=temulti(3) + temulti(3, resid=True), static_version="v2", ref=REF)
run("lgb_m3_resid_only", full, te_specs=temulti(3, resid=True), static_version="v2", ref=REF)
