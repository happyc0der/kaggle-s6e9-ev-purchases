from __future__ import annotations

import subprocess
import time

import numpy as np
import pandas as pd

from .config import COMP, ID, SUB, TARGET, EXP
from .data import load


def write_submission(pred: np.ndarray, name: str, cv: float | None = None) -> str:
    te = load("test")
    assert len(pred) == len(te)
    path = SUB / f"{name}.csv"
    pd.DataFrame({ID: te[ID].to_numpy(), TARGET: pred}).to_csv(path, index=False)
    with open(EXP / "submissions_log.csv", "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M')},{name},{cv if cv is not None else ''}\n")
    return str(path)


def kaggle_submit(path: str, message: str):
    return subprocess.run(["kaggle", "competitions", "submit", "-c", COMP, "-f", path, "-m", message],
                          capture_output=True, text=True)
