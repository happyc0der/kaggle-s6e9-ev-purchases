import numpy as np
import pandas as pd

INC = "Annual_Income_USD"


def hard_edges(pred: np.ndarray, df: pd.DataFrame) -> np.ndarray:
    """Deterministic regions found in train: income >= 170,537 always Yes; [31,004, 41,970] always No."""
    p = pred.copy()
    inc = df[INC].to_numpy(float)
    p[inc >= 170_537] = 1.0
    p[(inc >= 31_004) & (inc <= 41_970)] = 0.0
    return p
