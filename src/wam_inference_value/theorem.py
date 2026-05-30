"""Exact finite inference laws for top-score rollout selection.

The source of truth in this project is the finite empirical law. Given a pool
of sampled rollouts, top-score best-of-N selection is an order statistic over
that empirical distribution. Ties are handled by uniform random tie-breaking.

For a real-valued rollout utility R and score S, the expected selected utility
is the sum over score tie groups g:

    mean_R_g * [(r_max_g / m)^N - ((r_min_g - 1) / m)^N]

where m is the pool size and ranks are 1-indexed after sorting scores in
ascending order. Binary success is the special case R in {0, 1}.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class TieGroup:
    """A score tie group in ascending score order."""

    score: float
    start: int
    stop: int
    r_min: int
    r_max: int

    @property
    def size(self) -> int:
        return self.stop - self.start


def _as_1d_float(values: Iterable[float], name: str) -> np.ndarray:
    arr = np.asarray(list(values) if not isinstance(values, np.ndarray) else values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def _as_n_values(n_values: Iterable[int]) -> list[int]:
    out = [int(n) for n in n_values]
    if not out:
        raise ValueError("n_values must be non-empty")
    if any(n < 1 for n in out):
        raise ValueError("all N values must be >= 1")
    return out


def _sorted_groups(scores: np.ndarray) -> tuple[np.ndarray, list[TieGroup]]:
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    groups: list[TieGroup] = []
    i = 0
    n = len(sorted_scores)
    while i < n:
        j = i + 1
        while j < n and sorted_scores[j] == sorted_scores[i]:
            j += 1
        groups.append(
            TieGroup(
                score=float(sorted_scores[i]),
                start=i,
                stop=j,
                r_min=i + 1,
                r_max=j,
            )
        )
        i = j
    return order, groups


def utility_best_of_n_finite(
    scores: Iterable[float],
    utilities: Iterable[float],
    n_values: Iterable[int],
) -> dict[int, float]:
    """Expected utility of top-score best-of-N from a finite rollout pool.

    Sampling is i.i.d. with replacement from the pool. If the maximum score is
    tied, the selected rollout is chosen uniformly from the tied top-score
    samples, which is equivalent to using the tie group's mean utility.
    """

    scores_arr = _as_1d_float(scores, "scores")
    utilities_arr = _as_1d_float(utilities, "utilities")
    if scores_arr.shape != utilities_arr.shape:
        raise ValueError("scores and utilities must have the same length")
    ns = _as_n_values(n_values)

    n = len(scores_arr)
    order, groups = _sorted_groups(scores_arr)
    sorted_utilities = utilities_arr[order]
    out: dict[int, float] = {}
    for N in ns:
        val = 0.0
        for group in groups:
            mass = (group.r_max / n) ** N - ((group.r_min - 1) / n) ** N
            val += float(np.mean(sorted_utilities[group.start : group.stop])) * mass
        out[N] = float(val)
    return out


def binary_best_of_n_finite(
    scores: Iterable[float],
    success: Iterable[bool | int | float],
    n_values: Iterable[int],
) -> dict[int, float]:
    """Exact finite best-of-N success probability."""

    success_arr = _as_1d_float(success, "success")
    if not np.all((success_arr == 0.0) | (success_arr == 1.0)):
        raise ValueError("success must be binary")
    return utility_best_of_n_finite(scores, success_arr, n_values)


def binary_best_of_n_continuous_estimate(
    scores: Iterable[float],
    success: Iterable[bool | int | float],
    n_values: Iterable[int],
) -> dict[int, float]:
    """Plug-in estimate of f_N = N p E[F_mix(S+)^(N-1)].

    This is useful as a continuous-distribution diagnostic. It is not the
    tie-aware source of truth for finite rollout pools.
    """

    scores_arr = _as_1d_float(scores, "scores")
    success_arr = _as_1d_float(success, "success")
    if scores_arr.shape != success_arr.shape:
        raise ValueError("scores and success must have the same length")
    if not np.all((success_arr == 0.0) | (success_arr == 1.0)):
        raise ValueError("success must be binary")

    ns = _as_n_values(n_values)
    p = float(np.mean(success_arr))
    if p == 0.0:
        return {N: 0.0 for N in ns}
    sorted_scores = np.sort(scores_arr, kind="mergesort")
    pos_scores = scores_arr[success_arr == 1.0]
    u = np.searchsorted(sorted_scores, pos_scores, side="right") / len(scores_arr)
    return {N: float(N * p * np.mean(u ** (N - 1))) for N in ns}


def auc_kappa(scores: Iterable[float], success: Iterable[bool | int | float]) -> float:
    """Tie-aware AUC/kappa: P(S+ > S-) + 0.5 P(S+ = S-)."""

    scores_arr = _as_1d_float(scores, "scores")
    success_arr = _as_1d_float(success, "success")
    if scores_arr.shape != success_arr.shape:
        raise ValueError("scores and success must have the same length")
    if not np.all((success_arr == 0.0) | (success_arr == 1.0)):
        raise ValueError("success must be binary")
    pos = scores_arr[success_arr == 1.0]
    neg = scores_arr[success_arr == 0.0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    wins = 0.0
    for s in pos:
        wins += float(np.sum(s > neg)) + 0.5 * float(np.sum(s == neg))
    return float(wins / (len(pos) * len(neg)))


def n2_auc_identity(p: float, kappa: float) -> float:
    """N=2 identity f_2 = p^2 + 2p(1-p)kappa."""

    p = float(p)
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    if not np.isfinite(kappa):
        raise ValueError("kappa must be finite when 0 < p < 1")
    return float(p * p + 2.0 * p * (1.0 - p) * float(kappa))


def rank_interval_moments(
    scores: Iterable[float],
    success: Iterable[bool | int | float],
    n_values: Iterable[int],
) -> dict[int, float]:
    """Finite empirical theta moments satisfying f_N = N p theta_{N-1}."""

    success_arr = _as_1d_float(success, "success")
    p = float(np.mean(success_arr))
    ns = _as_n_values(n_values)
    if p == 0.0:
        return {N: 0.0 for N in ns}
    curve = binary_best_of_n_finite(scores, success_arr, ns)
    return {N: float(curve[N] / (N * p)) for N in ns}


def tie_rate(scores: Iterable[float]) -> float:
    """Fraction of score pairs that are tied."""

    scores_arr = _as_1d_float(scores, "scores")
    n = len(scores_arr)
    if n < 2:
        return 0.0
    _, counts = np.unique(scores_arr, return_counts=True)
    tied_pairs = sum(int(c) * (int(c) - 1) / 2 for c in counts if c > 1)
    return float(tied_pairs / (n * (n - 1) / 2))


def simulate_best_of_n(
    scores: Iterable[float],
    values: Iterable[float],
    N: int,
    n_trials: int = 10_000,
    seed: int | None = None,
) -> float:
    """Monte Carlo sanity check for best-of-N selected value."""

    scores_arr = _as_1d_float(scores, "scores")
    values_arr = _as_1d_float(values, "values")
    if scores_arr.shape != values_arr.shape:
        raise ValueError("scores and values must have the same length")
    if N < 1:
        raise ValueError("N must be >= 1")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(scores_arr), size=(int(n_trials), int(N)))
    chosen_scores = scores_arr[idx]
    if len(np.unique(scores_arr)) == len(scores_arr):
        best_per_trial = np.argmax(chosen_scores, axis=1)
        selected = idx[np.arange(int(n_trials)), best_per_trial]
        return float(np.mean(values_arr[selected]))
    max_scores = np.max(chosen_scores, axis=1)
    selected = np.empty(int(n_trials), dtype=int)
    for row_idx in range(int(n_trials)):
        tied_positions = np.flatnonzero(chosen_scores[row_idx] == max_scores[row_idx])
        selected_position = rng.choice(tied_positions)
        selected[row_idx] = idx[row_idx, selected_position]
    return float(np.mean(values_arr[selected]))


def auc_only_constant_moment_curve(
    p: float,
    kappa: float,
    n_values: Iterable[int],
) -> dict[int, float]:
    """A deliberately limited AUC-only high-N baseline.

    It uses the exact N=2 identity to infer theta_1 and then assumes all higher
    moments equal theta_1^(N-1). The experiment uses this as a foil: it is exact
    for N=2 and generally wrong for upper-tail best-of-N behavior.
    """

    ns = _as_n_values(n_values)
    p = float(p)
    if p <= 0.0:
        return {N: 0.0 for N in ns}
    if p >= 1.0:
        return {N: 1.0 for N in ns}
    theta1 = n2_auc_identity(p, kappa) / (2.0 * p)
    out = {}
    for N in ns:
        if N == 1:
            pred = p
        elif N == 2:
            pred = n2_auc_identity(p, kappa)
        else:
            pred = N * p * (theta1 ** (N - 1))
        out[N] = float(np.clip(pred, 0.0, 1.0))
    return out
