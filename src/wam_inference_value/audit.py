"""Inference-value audit utilities.

The theorem gives the exact value curve for a fixed score/utility
distribution. This module turns that curve into deployment diagnostics:
profile class, tail alignment, stop rules, and a conservative gate for whether
high-N imagination should be trusted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from wam_inference_value.stats import normalized_utility
from wam_inference_value.theorem import utility_best_of_n_finite


@dataclass(frozen=True)
class AuditDecision:
    """Conservative action suggested by an inference-value audit."""

    action: str
    reason: str
    recommended_n: int


def _as_float_array(values: Iterable[float], name: str) -> np.ndarray:
    arr = np.asarray(list(values) if not isinstance(values, np.ndarray) else values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def _rankdata_average(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and sorted_values[j] == sorted_values[i]:
            j += 1
        avg_rank = 0.5 * (i + j - 1)
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return 0.0
    aa = a - float(np.mean(a))
    bb = b - float(np.mean(b))
    denom = float(np.sqrt(np.sum(aa * aa) * np.sum(bb * bb)))
    if denom <= 1e-12:
        return 0.0
    return float(np.sum(aa * bb) / denom)


def inference_value_profile(
    scores: Iterable[float],
    utilities: Iterable[float],
    n_values: Iterable[int],
    *,
    normalize: bool = True,
) -> dict:
    """Return exact best-of-N utility curve and marginal gains."""

    scores_arr = _as_float_array(scores, "scores")
    utilities_arr = _as_float_array(utilities, "utilities")
    if scores_arr.shape != utilities_arr.shape:
        raise ValueError("scores and utilities must have the same length")
    n_grid = [int(n) for n in n_values]
    if not n_grid or any(n < 1 for n in n_grid):
        raise ValueError("n_values must be non-empty and >= 1")
    value_utilities = normalized_utility(utilities_arr) if normalize else utilities_arr
    curve = utility_best_of_n_finite(scores_arr, value_utilities, n_grid)
    ordered = sorted(curve.items())
    marginal_rows = []
    previous_n = None
    previous_v = None
    for n, value in ordered:
        if previous_n is None:
            marginal_rows.append({"N": n, "value": value, "delta_value": 0.0, "delta_per_rollout": 0.0})
        else:
            delta_n = max(1, n - previous_n)
            delta = value - previous_v
            marginal_rows.append(
                {
                    "N": n,
                    "value": value,
                    "delta_value": float(delta),
                    "delta_per_rollout": float(delta / delta_n),
                }
            )
        previous_n = n
        previous_v = value
    values = np.asarray([v for _, v in ordered], dtype=float)
    first = float(values[0])
    last = float(values[-1])
    best = float(np.max(values))
    worst_drop = float(np.min(np.diff(values))) if len(values) > 1 else 0.0
    return {
        "normalized": bool(normalize),
        "curve": {int(n): float(v) for n, v in ordered},
        "rows": marginal_rows,
        "n_min": int(ordered[0][0]),
        "n_max": int(ordered[-1][0]),
        "value_first": first,
        "value_last": last,
        "value_best": best,
        "gain_last_minus_first": last - first,
        "best_minus_first": best - first,
        "worst_adjacent_delta": worst_drop,
    }


def classify_profile(profile: dict, *, gain_threshold: float = 0.05, harm_threshold: float = 0.02) -> str:
    """Classify an inference-value profile on a normalized utility scale."""

    gain = float(profile["gain_last_minus_first"])
    best_gain = float(profile["best_minus_first"])
    worst_delta = float(profile["worst_adjacent_delta"])
    if gain < -harm_threshold:
        return "harmful"
    if worst_delta < -harm_threshold and best_gain > gain_threshold:
        return "unstable"
    if gain < gain_threshold:
        return "saturating"
    return "helpful"


def tail_alignment(
    scores: Iterable[float],
    real_utilities: Iterable[float],
    imagined_utilities: Iterable[float] | None = None,
    *,
    top_fraction: float = 0.10,
) -> dict:
    """Measure whether the high-score tail is aligned with real utility."""

    scores_arr = _as_float_array(scores, "scores")
    real_arr = normalized_utility(_as_float_array(real_utilities, "real_utilities"))
    if scores_arr.shape != real_arr.shape:
        raise ValueError("scores and real_utilities must have the same length")
    imag_arr = None
    if imagined_utilities is not None:
        imag_arr = normalized_utility(_as_float_array(imagined_utilities, "imagined_utilities"))
        if imag_arr.shape != real_arr.shape:
            raise ValueError("imagined_utilities must have the same length as real_utilities")

    k = max(1, int(np.ceil(len(scores_arr) * float(top_fraction))))
    top_idx = np.argsort(scores_arr, kind="mergesort")[-k:]
    real_mean = float(np.mean(real_arr))
    real_tail = float(np.mean(real_arr[top_idx]))
    score_real_corr = _corr(scores_arr, real_arr)
    score_real_rank_corr = _corr(_rankdata_average(scores_arr), _rankdata_average(real_arr))
    out = {
        "top_fraction": float(top_fraction),
        "top_count": int(k),
        "score_real_corr": score_real_corr,
        "score_real_rank_corr": score_real_rank_corr,
        "tail_real_mean": real_tail,
        "tail_real_uplift": real_tail - real_mean,
    }
    if imag_arr is not None:
        imag_mean = float(np.mean(imag_arr))
        imag_tail = float(np.mean(imag_arr[top_idx]))
        out.update(
            {
                "score_imagined_corr": _corr(scores_arr, imag_arr),
                "real_imagined_corr": _corr(real_arr, imag_arr),
                "tail_imagined_mean": imag_tail,
                "tail_imagined_uplift": imag_tail - imag_mean,
                "tail_hallucination_gap": imag_tail - real_tail,
            }
        )
    status = "weak"
    if out["tail_real_uplift"] < -0.05 or out["score_real_rank_corr"] < -0.20:
        status = "anti_aligned"
    elif out.get("tail_hallucination_gap", 0.0) > 0.20 and out["tail_real_uplift"] < 0.08:
        status = "hallucinated_tail"
    elif out["tail_real_uplift"] > 0.08 and out["score_real_rank_corr"] > 0.15:
        status = "aligned"
    out["alignment_status"] = status
    return out


def stop_rule(profile: dict, *, compute_cost_per_rollout: float = 0.0015, patience: int = 2) -> int:
    """Pick the earliest N after marginal value stays below compute cost."""

    rows = profile["rows"]
    below = 0
    for i, row in enumerate(rows[1:], start=1):
        if row["delta_per_rollout"] <= compute_cost_per_rollout:
            below += 1
            if below >= patience:
                return int(rows[max(0, i - patience + 1)]["N"])
        else:
            below = 0
    return int(profile["n_max"])


def deployment_gate(profile_class: str, alignment: dict, stop_n: int, n_max: int) -> AuditDecision:
    """Convert profile diagnostics into a conservative deployment action."""

    align = alignment.get("alignment_status", "weak")
    if profile_class == "harmful" or align in {"anti_aligned", "hallucinated_tail"}:
        return AuditDecision(
            action="block_high_n",
            reason=f"profile={profile_class}, alignment={align}",
            recommended_n=1,
        )
    if stop_n < n_max:
        return AuditDecision(
            action="stop_early",
            reason="marginal inference value below compute threshold",
            recommended_n=int(stop_n),
        )
    if profile_class == "helpful" and align == "aligned":
        return AuditDecision(action="sample_more", reason="high-score tail is real-utility aligned", recommended_n=int(n_max))
    return AuditDecision(action="pilot_more", reason=f"profile={profile_class}, alignment={align}", recommended_n=int(stop_n))


def audit_score_distribution(
    scores: Iterable[float],
    real_utilities: Iterable[float],
    n_values: Iterable[int],
    *,
    imagined_utilities: Iterable[float] | None = None,
    top_fraction: float = 0.10,
    compute_cost_per_rollout: float = 0.0015,
) -> dict:
    """Full inference-value audit for one score/utility distribution."""

    profile = inference_value_profile(scores, real_utilities, n_values, normalize=True)
    profile_class = classify_profile(profile)
    alignment = tail_alignment(scores, real_utilities, imagined_utilities, top_fraction=top_fraction)
    stop_n = stop_rule(profile, compute_cost_per_rollout=compute_cost_per_rollout)
    decision = deployment_gate(profile_class, alignment, stop_n, int(profile["n_max"]))
    return {
        "profile": profile,
        "profile_class": profile_class,
        "alignment": alignment,
        "stop_n": int(stop_n),
        "decision": decision.__dict__,
    }
