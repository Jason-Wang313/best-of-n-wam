"""Shared experiment helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from wam_inference_value.scorers import scores_for_pool
from wam_inference_value.theorem import (
    auc_kappa,
    binary_best_of_n_finite,
    tie_rate,
    utility_best_of_n_finite,
)


N_VALUES = [1, 2, 4, 8, 16, 32, 64]


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def results_dir() -> Path:
    return project_root() / "results"


def ensure_result_dirs() -> None:
    for path in [results_dir(), results_dir() / "tables", results_dir() / "figures"]:
        path.mkdir(parents=True, exist_ok=True)


def json_sanitize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_sanitize(v) for v in obj]
    if isinstance(obj, tuple):
        return [json_sanitize(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return json_sanitize(obj.tolist())
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_sanitize(payload), indent=2), encoding="utf-8")


def normalize_values(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values
    lo = float(np.min(values))
    hi = float(np.max(values))
    if hi - lo <= 1e-12:
        return np.full_like(values, 0.5, dtype=float)
    return (values - lo) / (hi - lo)


def ci95(values: list[float] | np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"n": 0, "mean": None, "std": None, "stderr": None, "ci95": None, "lo": None, "hi": None}
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    stderr = float(std / np.sqrt(arr.size)) if arr.size > 1 else 0.0
    half_width = float(1.96 * stderr)
    return {
        "n": int(arr.size),
        "mean": mean,
        "std": std,
        "stderr": stderr,
        "ci95": half_width,
        "lo": mean - half_width,
        "hi": mean + half_width,
    }


def seed_list_from_args(args: Any) -> list[int]:
    explicit = getattr(args, "seeds", None)
    if explicit:
        return [int(s) for s in explicit]
    base = int(getattr(args, "seed", 0))
    num = int(getattr(args, "num_seeds", 1))
    return [base + 10_007 * i for i in range(max(1, num))]


def backend_suffix(args: Any) -> str:
    backend = getattr(args, "dynamics_backend", "analytic")
    return "" if backend == "analytic" else f"_{backend}"


def load_model_for_backend(args: Any):
    backend = getattr(args, "dynamics_backend", "analytic")
    if backend != "learned":
        return None
    from wam_inference_value.learned_wam import load_or_train_learned_wam_lite

    model_path = getattr(args, "model_path", None) or str(results_dir() / "models" / "learned_wam_lite.npz")
    return load_or_train_learned_wam_lite(
        model_path=model_path,
        train_if_missing=bool(getattr(args, "train_if_missing", False)),
        seed=int(getattr(args, "model_seed", getattr(args, "seed", 101))),
        id_mismatch=str(getattr(args, "id_mismatch", "mild")),
        train_states=int(getattr(args, "train_states", 64)),
        train_rollouts=int(getattr(args, "train_rollouts", 96)),
        val_states=int(getattr(args, "val_states", 24)),
        val_rollouts=int(getattr(args, "val_rollouts", 96)),
        max_horizon=int(getattr(args, "max_horizon", 12)),
    )


def add_backend_args(parser: Any) -> None:
    parser.add_argument("--dynamics-backend", choices=["analytic", "learned", "oracle_true"], default="analytic")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--train-if-missing", action="store_true")
    parser.add_argument("--model-seed", type=int, default=101)
    parser.add_argument("--id-mismatch", type=str, default="mild")
    parser.add_argument("--train-states", type=int, default=64)
    parser.add_argument("--train-rollouts", type=int, default=96)
    parser.add_argument("--val-states", type=int, default=24)
    parser.add_argument("--val-rollouts", type=int, default=96)
    parser.add_argument("--max-horizon", type=int, default=12)
    parser.add_argument("--num-seeds", type=int, default=1)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)


def curve_rows_for_pool(pool, scorer: str, n_values: list[int] | None = None) -> list[dict[str, Any]]:
    n_values = N_VALUES if n_values is None else n_values
    scores = scores_for_pool(pool, scorer)
    success = pool.real_success
    real_utility = pool.real_utility
    imagined_utility = pool.imagined_utility
    normalized_real_utility = normalize_values(real_utility)
    normalized_imagined_utility = normalize_values(imagined_utility)
    success_curve = binary_best_of_n_finite(scores, success, n_values)
    utility_curve = utility_best_of_n_finite(scores, real_utility, n_values)
    imagined_curve = utility_best_of_n_finite(scores, imagined_utility, n_values)
    normalized_utility_curve = utility_best_of_n_finite(scores, normalized_real_utility, n_values)
    normalized_imagined_curve = utility_best_of_n_finite(scores, normalized_imagined_utility, n_values)
    p = float(np.mean(success))
    kappa = auc_kappa(scores, success)
    rows = []
    for N in n_values:
        rows.append(
            {
                "state_id": pool.state_id,
                "mismatch": pool.mismatch,
                "scorer": scorer,
                "N": int(N),
                "success": success_curve[N],
                "real_utility": utility_curve[N],
                "imagined_utility": imagined_curve[N],
                "gap_imagined_minus_real": imagined_curve[N] - utility_curve[N],
                "normalized_real_utility": normalized_utility_curve[N],
                "normalized_imagined_utility": normalized_imagined_curve[N],
                "normalized_gap_imagined_minus_real": normalized_imagined_curve[N] - normalized_utility_curve[N],
                "p": p,
                "kappa": kappa,
                "tie_rate": tie_rate(scores),
            }
        )
    return rows


def aggregate_curve_table(rows: list[dict[str, Any]], group_cols: list[str]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    agg_cols = [
        "success",
        "real_utility",
        "imagined_utility",
        "gap_imagined_minus_real",
        "normalized_real_utility",
        "normalized_imagined_utility",
        "normalized_gap_imagined_minus_real",
        "p",
        "kappa",
        "tie_rate",
    ]
    present = [c for c in agg_cols if c in df.columns]
    return df.groupby(group_cols, dropna=False)[present].mean().reset_index()


def area_under_inference_curve(df: pd.DataFrame, value_col: str = "real_utility") -> float:
    sub = df.sort_values("N")
    x = np.asarray(sub["N"], dtype=float)
    y = np.asarray(sub[value_col], dtype=float)
    if len(x) < 2:
        return float(y[0]) if len(y) else float("nan")
    return float(np.trapz(y, x) / (x[-1] - x[0]))


def marginal_greedy_allocate(pred_curves: list[np.ndarray], total_budget: int) -> list[int]:
    n_items = len(pred_curves)
    alloc = np.ones(n_items, dtype=int)
    maxes = np.asarray([len(c) for c in pred_curves], dtype=int)
    remaining = max(0, int(total_budget) - n_items)
    while remaining > 0:
        gains = np.full(n_items, -np.inf, dtype=float)
        for i, curve in enumerate(pred_curves):
            if alloc[i] < maxes[i]:
                gains[i] = curve[alloc[i]] - curve[alloc[i] - 1]
        best = int(np.argmax(gains))
        if not np.isfinite(gains[best]) or gains[best] <= 1e-12:
            break
        alloc[best] += 1
        remaining -= 1
    return alloc.tolist()


def uniform_allocate(n_items: int, total_budget: int, max_n: int) -> list[int]:
    alloc = np.ones(n_items, dtype=int)
    remaining = max(0, int(total_budget) - n_items)
    idx = 0
    while remaining > 0 and np.any(alloc < max_n):
        if alloc[idx] < max_n:
            alloc[idx] += 1
            remaining -= 1
        idx = (idx + 1) % n_items
    return alloc.tolist()
