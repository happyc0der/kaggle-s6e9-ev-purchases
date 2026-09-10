"""MLP with entity embeddings for categoricals + per-value ids of income/commute. Runs on MPS/CPU."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from ..config import CAT_COLS

INC, COM = "Annual_Income_USD", "Daily_Commute_km"
DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

NN_DEFAULT = dict(extra_cats=(), epochs=14, bs=4096, lr=2e-3, wd=1e-5, hidden=(512, 256, 128), drop=0.15, emb_inc=24, emb_com=12,
                  emb_cat=4, min_count=3, seed=0, patience=4)


class Net(nn.Module):
    def __init__(self, n_num, cat_cards, cat_dim, n_inc, d_inc, n_com, d_com, hidden, drop):
        super().__init__()
        self.cat_embs = nn.ModuleList([nn.Embedding(c, cat_dim) for c in cat_cards])
        self.inc_emb = nn.Embedding(n_inc, d_inc)
        self.com_emb = nn.Embedding(n_com, d_com)
        d = n_num + cat_dim * len(cat_cards) + d_inc + d_com
        layers = []
        for h in hidden:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.SiLU(), nn.Dropout(drop)]
            d = h
        layers += [nn.Linear(d, 1)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, xnum, xcat, xinc, xcom):
        parts = [xnum] + [e(xcat[:, i]) for i, e in enumerate(self.cat_embs)] + [self.inc_emb(xinc), self.com_emb(xcom)]
        return self.mlp(torch.cat(parts, 1)).squeeze(1)


def _vocab(values_all: np.ndarray, min_count: int):
    k = np.round(values_all * 10).astype(np.int64)
    u, c = np.unique(k, return_counts=True)
    keep = u[c >= min_count]
    return keep  # id = searchsorted index + 1; 0 = rare/unknown


def _ids(values: np.ndarray, vocab: np.ndarray):
    k = np.round(values * 10).astype(np.int64)
    if len(vocab) == 0:
        return np.zeros(len(k), dtype=np.int64)
    idx = np.searchsorted(vocab, k)
    idx_c = np.minimum(idx, len(vocab) - 1)
    return np.where(vocab[idx_c] == k, idx_c + 1, 0).astype(np.int64)


def _prep_num(Xtr: pd.DataFrame, others, num_cols):
    A = Xtr[num_cols].to_numpy(np.float32)
    logmask = (A.min(0) >= 0) & (A.max(0) > 50)
    def tf(X):
        B = X[num_cols].to_numpy(np.float32).copy()
        B[:, logmask] = np.log1p(B[:, logmask])
        return B
    A = tf(Xtr)
    mu, sd = A.mean(0), A.std(0) + 1e-6
    return [(tf(X) - mu) / sd for X in (Xtr, *others)]


def fit_nn(Xtr, ytr, Xva, yva, Xte, params=None, cat_cols=None):
    p = {**NN_DEFAULT, **(params or {})}
    torch.manual_seed(p["seed"]); np.random.seed(p["seed"])
    cats = [c for c in CAT_COLS + list(p.get("extra_cats", ())) if c in Xtr.columns]
    num_cols = [c for c in Xtr.columns if c not in cats]
    all_inc = np.concatenate([X[INC].to_numpy(float) for X in (Xtr, Xva, Xte)])
    all_com = np.concatenate([X[COM].to_numpy(float) for X in (Xtr, Xva, Xte)])
    v_inc, v_com = _vocab(all_inc, p["min_count"]), _vocab(all_com, p["min_count"])
    nums = _prep_num(Xtr, (Xva, Xte), num_cols)
    cat_cards = [int(max(X[c].max() for X in (Xtr, Xva, Xte))) + 1 for c in cats]

    def tens(X, num):
        return (torch.tensor(num), torch.tensor(X[cats].to_numpy(np.int64)) if cats else torch.zeros((len(X), 0), dtype=torch.long),
                torch.tensor(_ids(X[INC].to_numpy(float), v_inc)), torch.tensor(_ids(X[COM].to_numpy(float), v_com)))

    T = [tens(X, n) for X, n in zip((Xtr, Xva, Xte), nums)]
    T = [[t.to(DEV) for t in ts] for ts in T]
    ytr_t = torch.tensor(ytr, dtype=torch.float32, device=DEV)
    net = Net(len(num_cols), cat_cards, p["emb_cat"], len(v_inc) + 1, p["emb_inc"], len(v_com) + 1, p["emb_com"],
              p["hidden"], p["drop"]).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=p["lr"], weight_decay=p["wd"])
    n = len(ytr); steps = (n + p["bs"] - 1) // p["bs"]
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=p["lr"], total_steps=steps * p["epochs"], pct_start=0.2)
    lossf = nn.BCEWithLogitsLoss()

    def predict(ts):
        net.eval(); out = []
        with torch.no_grad():
            for i in range(0, len(ts[0]), 16384):
                out.append(torch.sigmoid(net(*[t[i:i + 16384] for t in ts])).float().cpu().numpy())
        return np.concatenate(out)

    best, best_ep, best_va, best_te = -1, -1, None, None
    for ep in range(p["epochs"]):
        net.train()
        perm = torch.randperm(n, device=DEV)
        for i in range(steps):
            b = perm[i * p["bs"]:(i + 1) * p["bs"]]
            opt.zero_grad()
            loss = lossf(net(*[t[b] for t in T[0]]), ytr_t[b])
            loss.backward(); opt.step(); sched.step()
        pva = predict(T[1]); auc = roc_auc_score(yva, pva)
        if auc > best:
            best, best_ep, best_va, best_te = auc, ep, pva, predict(T[2])
        elif ep - best_ep >= p["patience"]:
            break
    return best_va, best_te, best_ep
