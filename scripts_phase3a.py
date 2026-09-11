"""Phase 3a: class-balanced pseudo-labelling from the current blend's test predictions."""
import numpy as np, json
from scipy.special import logit, expit
from src.cv import run, load_oof, EXP
from src.blend import collect, optimize_weights
from src.features.static import GROUPS
from src.features.target_enc import TESpec
INC, COM = "Annual_Income_USD", "Daily_Commute_km"
full = GROUPS["raw"] + GROUPS["inc_stats"] + GROUPS["com_stats"] + GROUPS["digits"]
def temulti(m=3):
    return [TESpec((INC,), m=m), TESpec((COM,), m=m), TESpec((INC,), m=m, binwidth=100), TESpec((INC,), m=m, binwidth=500),
            TESpec((INC,), m=m, binwidth=2000), TESpec((COM,), m=m, binwidth=1), TESpec((COM,), m=m, binwidth=5)]
# blend test prediction (logit space) as the pseudo-label source
names = ["lgb_m3_tebag3", "xgb_m3_tebag3", "nn_E9_avg5"]
O, T, y = collect(names)
L, LT = logit(np.clip(O, 1e-6, 1 - 1e-6)), logit(np.clip(T, 1e-6, 1 - 1e-6))
w, auc = optimize_weights(L, y)
tp = expit(LT @ w)
d = EXP / "blend_src"; d.mkdir(exist_ok=True); np.save(d / "test.npy", tp)
print("pseudo source blend OOF", round(auc, 6))
REF = "lgb_full_temulti_m3"
# keep train positive rate 17.46%: frac_pos/(frac_pos+frac_neg) = 0.1746
run("lgb_m3_pl_small", full, te_specs=temulti(3), static_version="v2", pseudo=("blend_src", 0.0524, 0.2476), ref=REF)
run("lgb_m3_pl_large", full, te_specs=temulti(3), static_version="v2", pseudo=("blend_src", 0.1047, 0.4953), ref=REF)
