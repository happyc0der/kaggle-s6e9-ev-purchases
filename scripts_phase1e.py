"""Phase 1e: LightGBM native categoricals for income/commute, lower TE smoothing, staggered bins."""
from src.cv import run
from src.features.static import GROUPS
from src.features.target_enc import TESpec
INC, COM = "Annual_Income_USD", "Daily_Commute_km"
base = GROUPS["raw"] + GROUPS["inc_stats"] + GROUPS["com_stats"]
full = base + GROUPS["digits"]
def temulti(m=3, **kw):
    return [TESpec((INC,), m=m, **kw), TESpec((COM,), m=m, **kw),
            TESpec((INC,), m=m, binwidth=100, **kw), TESpec((INC,), m=m, binwidth=500, **kw),
            TESpec((INC,), m=m, binwidth=2000, **kw), TESpec((COM,), m=m, binwidth=1, **kw), TESpec((COM,), m=m, binwidth=5, **kw)]
REF = "lgb_full_temulti_m3"
# native categoricals: add integer copies so the raw numeric columns stay available too
def catcopy(tr, te, static):
    import pandas as pd
    return pd.DataFrame({"inc_cat": static[INC].to_numpy(float), "com_cat": static[COM].to_numpy(float)})
run("lgb_m3_nativecat", full, te_specs=temulti(3), static_version="v2", extra_fn=catcopy, cat_cols=["inc_cat", "com_cat"],
    params={"cat_smooth": 10, "cat_l2": 10, "min_data_per_group": 50, "max_cat_threshold": 64}, ref=REF)
run("lgb_m1", full, te_specs=temulti(1), static_version="v2", ref=REF)
stag = temulti(3) + [TESpec((INC,), m=3, binwidth=500, offset=250), TESpec((INC,), m=3, binwidth=2000, offset=1000),
                     TESpec((INC,), m=3, binwidth=250), TESpec((INC,), m=3, binwidth=5000), TESpec((COM,), m=3, binwidth=2)]
run("lgb_m3_stag", full, te_specs=stag, static_version="v2", ref=REF)
