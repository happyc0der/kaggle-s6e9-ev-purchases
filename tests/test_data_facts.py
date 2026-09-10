"""Assert the public facts about the dataset; a failure means the download or encoding is wrong."""
import numpy as np
import pytest

from src.config import TARGET, RAW
from src.data import load

pytestmark = pytest.mark.skipif(not (RAW / "train.csv").exists(), reason="data not downloaded")


def test_shapes_and_target():
    tr, te = load("train"), load("test")
    assert len(tr) == 668_665 and len(te) == 286_571
    assert te["id"].min() == 668_665
    assert abs(tr[TARGET].mean() - 0.1746) < 0.001
    assert tr.isna().sum().sum() == 0 and te.isna().sum().sum() == 0


def test_hard_edges():
    tr = load("train")
    inc, y = tr["Annual_Income_USD"].to_numpy(), tr[TARGET].to_numpy()
    hi = inc >= 170_537
    assert 390 <= hi.sum() <= 395 and y[hi].mean() == 1.0
    dead = (inc >= 31_004) & (inc <= 41_970)
    assert dead.sum() == 1257 and y[dead].sum() == 0
    assert abs((inc == 30_000).sum() - 61_606) <= 2
    assert (tr["Daily_Commute_km"] == 5.0).sum() == 144_280


def test_original():
    o = load("original")
    assert len(o) == 10_000
    assert o["Age"].min() == 25 and o["Age"].max() == 69


def test_te_leak_free():
    """Validation rows' encoding must not use their own target."""
    from src.features.target_enc import TESpec, nested_te
    tr = load("train").iloc[:20000].reset_index(drop=True)
    y = tr[TARGET].to_numpy(float)
    idx = np.arange(len(tr)); tr_idx, va_idx = idx[:15000], idx[15000:]
    spec = TESpec(("Annual_Income_USD",), m=1)
    _, enc_va, _ = nested_te(spec, tr, y, tr_idx, va_idx, tr.iloc[:5])
    # flipping validation labels must not change validation encodings
    y2 = y.copy(); y2[va_idx] = 1 - y2[va_idx]
    _, enc_va2, _ = nested_te(spec, tr, y2, tr_idx, va_idx, tr.iloc[:5])
    assert np.allclose(enc_va, enc_va2)
