"""Phase 1h: income residue (mod) features + TE on residues."""
import numpy as np, pandas as pd
from src.cv import run
from src.features.static import GROUPS
from src.features.target_enc import TESpec
INC, COM = "Annual_Income_USD", "Daily_Commute_km"
full = GROUPS["raw"] + GROUPS["inc_stats"] + GROUPS["com_stats"] + GROUPS["digits"]
def temulti(m=3):
    return [TESpec((INC,), m=m), TESpec((COM,), m=m), TESpec((INC,), m=m, binwidth=100), TESpec((INC,), m=m, binwidth=500),
            TESpec((INC,), m=m, binwidth=2000), TESpec((COM,), m=m, binwidth=1), TESpec((COM,), m=m, binwidth=5)]
def mods(tr, te, static):
    inc = np.round(static[INC].to_numpy(float)).astype(np.int64)
    return pd.DataFrame({"inc_res1000": inc % 1000, "inc_res500": inc % 500, "inc_res100": inc % 100})
REF = "lgb_full_temulti_m3"
run("lgb_m3_mods", full, te_specs=temulti(3), static_version="v2", extra_fn=mods, ref=REF)
