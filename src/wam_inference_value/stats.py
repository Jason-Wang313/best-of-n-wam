"""Statistical helpers for WAM experiment summaries."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from wam_inference_value.theorem import auc_kappa


def _finite_array(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values) if not isinstance(values, np.ndarray) else values, dtype=float)
    return arr[np.isfinite(arr)]


def bootstrap_ci(values: Iterable[float], *, seed: int = 0, n_boot: int = 2000, alpha: float = 0.05) -> dict:
    """Percentile bootstrap CI for the mean."""

    arr = _finite_array(values)
    if arr.size == 0:
        return {"n": 0, "mean": None, "lo": None, "hi": None, "std": None}
    if arr.size == 1:
        mean = float(arr[0])
        return {"n": 1, "mean": mean, "lo": mean, "hi": mean, "std": 0.0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(int(n_boot), arr.size))
    boot = np.mean(arr[idx], axis=1)
    lo, hi = np.quantile(boot, [alpha / 2.0, 1.0 - alpha / 2.0])
    return {"n": int(arr.size), "mean": float(np.mean(arr)), "lo": float(lo), "hi": float(hi), "std": float(np.std(arr, ddof=1))}


def paired_bootstrap_ci(
    a: Iterable[float],
    b: Iterable[float],
    *,
    seed: int = 0,
    n_boot: int = 2000,
    alpha: float = 0.05,
) -> dict:
    """Percentile bootstrap CI for mean(a - b)."""

    arr_a = np.asarray(list(a) if not isinstance(a, np.ndarray) else a, dtype=float)
    arr_b = np.asarray(list(b) if not isinstance(b, np.ndarray) else b, dtype=float)
    if arr_a.shape != arr_b.shape:
        raise ValueError("paired arrays must have the same shape")
    mask = np.isfinite(arr_a) & np.isfinite(arr_b)
    diff = arr_a[mask] - arr_b[mask]
    out = bootstrap_ci(diff, seed=seed, n_boot=n_boot, alpha=alpha)
    out["effect"] = out["mean"]
    return out


def effect_size(a: Iterable[float], b: Iterable[float]) -> float:
    """Cohen-style standardized paired mean difference."""

    arr_a = np.asarray(list(a) if not isinstance(a, np.ndarray) else a, dtype=float)
    arr_b = np.asarray(list(b) if not isinstance(b, np.ndarray) else b, dtype=float)
    if arr_a.shape != arr_b.shape:
        raise ValueError("paired arrays must have the same shape")
    diff = arr_a - arr_b
    diff = diff[np.isfinite(diff)]
    if diff.size == 0:
        return float("nan")
    std = float(np.std(diff, ddof=1)) if diff.size > 1 else 0.0
    if std <= 1e-12:
        return float("inf") if float(np.mean(diff)) > 0 else 0.0
    return float(np.mean(diff) / std)


def normalized_utility(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values) if not isinstance(values, np.ndarray) else values, dtype=float)
    if arr.size == 0:
        return arr
    lo = float(np.nanmin(arr))
    hi = float(np.nanmax(arr))
    if hi - lo <= 1e-12:
        return np.full_like(arr, 0.5, dtype=float)
    return (arr - lo) / (hi - lo)


def auc_with_ties(scores: Iterable[float], labels: Iterable[float]) -> float:
    return auc_kappa(scores, labels)


def claim_status_from_ci(ci: dict, *, threshold: float = 0.0) -> str:
    lo = ci.get("lo")
    mean = ci.get("mean")
    if lo is None or mean is None:
        return "UNSUPPORTED"
    if lo > threshold:
        return "VERIFIED"
    if mean > threshold:
        return "PARTIAL"
    return "FAILED"
