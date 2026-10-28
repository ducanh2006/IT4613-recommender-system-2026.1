from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy import stats


def bootstrap_ci(
    values: Sequence[float],
    n_boot: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    samples = rng.choice(array, size=(n_boot, len(array)), replace=True)
    estimates = samples.mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(estimates, [alpha, 1.0 - alpha])
    return float(low), float(high)


def paired_bootstrap_delta(
    values_a: Sequence[float],
    values_b: Sequence[float],
    n_boot: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    valid = np.isfinite(a) & np.isfinite(b)
    a = a[valid]
    b = b[valid]
    if len(a) == 0:
        return float("nan"), float("nan"), float("nan")
    differences = a - b
    observed = float(differences.mean())
    rng = np.random.default_rng(seed)
    samples = rng.choice(differences, size=(n_boot, len(differences)), replace=True)
    estimates = samples.mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(estimates, [alpha, 1.0 - alpha])
    return observed, float(low), float(high)


def wilcoxon_p(values_a: Sequence[float], values_b: Sequence[float]) -> float:
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    valid = np.isfinite(a) & np.isfinite(b)
    a = a[valid]
    b = b[valid]
    if len(a) == 0 or np.allclose(a, b):
        return 1.0
    try:
        result = stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
    except ValueError:
        return 1.0
    return float(result.pvalue)


def cliffs_delta(values_a: Sequence[float], values_b: Sequence[float]) -> float:
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return 0.0
    greater = 0.0
    lesser = 0.0
    for value in a:
        greater += float(np.sum(value > b))
        lesser += float(np.sum(value < b))
    return (greater - lesser) / (len(a) * len(b))


def holm_bonferroni(p_values: Sequence[float]) -> list[float]:
    values = np.asarray(p_values, dtype=float)
    if len(values) == 0:
        return []
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    total = len(values)
    for position, index in enumerate(order):
        candidate = (total - position) * values[index]
        running = max(running, candidate)
        adjusted[index] = min(running, 1.0)
    return adjusted.tolist()


def summarize_values(
    values: Sequence[float],
    n_boot: int = 1000,
    seed: int = 42,
) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "n": 0.0,
        }
    low, high = bootstrap_ci(array, n_boot=n_boot, seed=seed)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "ci_low": low,
        "ci_high": high,
        "n": float(len(array)),
    }
