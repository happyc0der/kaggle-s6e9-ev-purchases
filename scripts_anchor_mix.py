"""Mix the public anchor (no OOF) with our OOF-validated mega blend in rank space; boundary rules; lexsort tie-break."""
import sys, numpy as np, pandas as pd
from scipy.stats import rankdata
from src.data import load
from src.submit import write_submission
tr, te = load("train"), load("test"); y = tr.Will_Buy_EV.values
def rules(d):
    inc = d.Annual_Income_USD.to_numpy(float); km = d.Daily_Commute_km.to_numpy(float)
    env1 = d.Environmental_Concern_Level.to_numpy() == 1; no_sub = d.Subsidy_Available.to_numpy() == 0
    anx = d.Range_Anxiety_Level.to_numpy() >= 1
    s = np.zeros(len(d)); s[inc >= 170537] += 10; s[(inc >= 31004) & (inc <= 41970)] -= 10
    s[km >= 83] -= 5; s[(inc == 30000) & no_sub & (env1 | anx)] -= 5
    return s
s_tr = rules(tr)
for name, m in [("km>=83", s_tr == -5), ("30k&nosub&(env1|anx)", (tr.Annual_Income_USD.values == 30000) & (s_tr <= -5))]:
    print(f"{name}: rows {m.sum()} buyers {int(y[m].sum())}")
rk = lambda v: rankdata(v) / len(v)
anchor = pd.read_csv("data/public/zoomzoom/unz/own/own_001_56267408.csv").set_index("id").reindex(te.id)["Will_Buy_EV"].to_numpy(float)
ours = np.load("experiments/megablend_test.npy")
nn = pd.read_csv("data/public/sixviews/test_realmlp_g.csv").set_index("id").reindex(te.id)["G_realmlp_3seed"].to_numpy(float)
N = len(te)
def lexrank(primary, secondary):
    o = np.lexsort((secondary, primary)); r = np.empty(N); r[o] = np.arange(1, N + 1); return (r - 0.5) / N
from scipy.stats import spearmanr
print("spearman(anchor, ours):", round(spearmanr(anchor, ours).statistic, 5))
for tag, wa in [("mega2_a50", 0.5), ("mega3_a70", 0.7)]:
    blend = wa * rk(anchor) + (1 - wa) * rk(ours)
    final = lexrank(blend + rules(te), nn)
    assert len(np.unique(final)) == N
    path = write_submission(final, tag, cv=None); print("wrote", path)
