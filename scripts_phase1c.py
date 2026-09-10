"""Phase 1c: interaction target encodings and light tuning on the best LGBM set."""
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
REF = "lgb_full_temulti"

te_inter = te_multi + [TESpec((INC, "Subsidy_Available"), m=10), TESpec((INC, "Range_Anxiety_Level"), m=10),
                       TESpec((INC, "Environmental_Concern_Level"), m=10), TESpec((INC, COM), m=10)]
run("lgb_full_teinter", full, te_specs=te_inter, static_version="v2", ref=REF)
te_bininter = te_multi + [TESpec((INC, "Subsidy_Available"), m=10, binwidth=500),
                          TESpec((INC, "Range_Anxiety_Level"), m=10, binwidth=500),
                          TESpec((INC, "Environmental_Concern_Level"), m=10, binwidth=500)]
run("lgb_full_tebininter", full, te_specs=te_bininter, static_version="v2", ref=REF)
run("lgb_full_temulti_tuneA", full, te_specs=te_multi, static_version="v2",
    params={"learning_rate": 0.01, "num_leaves": 127, "min_child_samples": 200, "feature_fraction": 0.6}, ref=REF)
run("lgb_full_temulti_tuneB", full, te_specs=te_multi, static_version="v2",
    params={"learning_rate": 0.01, "num_leaves": 31, "min_child_samples": 50, "feature_fraction": 0.8, "lambda_l2": 10}, ref=REF)
