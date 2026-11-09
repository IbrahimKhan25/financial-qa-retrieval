"""Query-level bootstrap confidence intervals, plain and paired."""
import numpy as np


def _ci(samples: np.ndarray, alpha: float) -> tuple[float, float]:
    lo, hi = np.percentile(samples, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def bootstrap_ci(x, n_boot: int = 10_000, alpha: float = 0.05, seed: int = 42) -> dict:
    """Mean of per-query scores with a percentile bootstrap CI (resampling queries)."""
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    lo, hi = _ci(means, alpha)
    return {"mean": float(x.mean()), "ci_low": lo, "ci_high": hi}


def paired_bootstrap(a, b, n_boot: int = 10_000, alpha: float = 0.05, seed: int = 42) -> dict:
    """mean(b - a) over the same queries, with a CI and a two-sided bootstrap p-value.

    a and b must be aligned per query (same order). Resampling the same query indices for both
    systems keeps the pairing, which is why this is tighter than comparing two separate CIs.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("a and b must be aligned per query")
    d = b - a
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    lo, hi = _ci(means, alpha)
    # p-value: how often the resampled mean falls on the opposite side of zero (x2, capped at 1)
    p = min(1.0, 2 * min((means <= 0).mean(), (means >= 0).mean()))
    return {"diff": float(d.mean()), "ci_low": lo, "ci_high": hi, "p_value": float(p)}
