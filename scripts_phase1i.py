"""Phase 1i: target encodings keyed on income residues."""
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
    return pd.DataFrame({"inc_mod1000": inc % 1000, "inc_mod500": inc % 500, "inc_mod100": inc % 100})
REF = "lgb_full_temulti_m3"
te_mod = temulti(3) + [TESpec((INC,), m=3, modulus=1000), TESpec((INC,), m=3, modulus=100), TESpec((INC,), m=3, modulus=10)]
run("lgb_m3_mods_temod", full, te_specs=te_mod, static_version="v2", extra_fn=mods, ref="lgb_m3_mods")
