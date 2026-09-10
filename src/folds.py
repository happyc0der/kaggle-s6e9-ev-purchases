import numpy as np
from sklearn.model_selection import StratifiedKFold

from .config import FOLD_SEED, N_FOLDS, PROC


def get_folds(y: np.ndarray, n_folds: int = N_FOLDS, seed: int = FOLD_SEED) -> np.ndarray:
    """Frozen fold id per train row, saved once to disk so every experiment shares it."""
    path = PROC / f"folds_k{n_folds}_s{seed}.npy"
    if path.exists():
        f = np.load(path)
        assert len(f) == len(y)
        return f
    f = np.full(len(y), -1, dtype=np.int8)
    skf = StratifiedKFold(n_folds, shuffle=True, random_state=seed)
    for k, (_, va) in enumerate(skf.split(np.zeros(len(y)), y)):
        f[va] = k
    np.save(path, f)
    return f
