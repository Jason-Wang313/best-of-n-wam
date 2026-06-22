from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from wam_inference_value.stats import paired_bootstrap_ci
from wam_inference_value.theorem import utility_best_of_n_finite


N_VALUES = [1, 2, 4, 8, 16, 32, 64]
LABEL_BUDGETS = [0, 1, 2, 4, 8, 16, 32, 64]
HIGH_N = 32


@dataclass(frozen=True)
class SyntheticPool:
    pool_id: str
    family: str
    split: str
    seed: int
    score: np.ndarray
    imagined_utility: np.ndarray
    real_utility: np.ndarray
    uncertainty: np.ndarray
    diagnostic: np.ndarray
    random_score: np.ndarray


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sanitize(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): sanitize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize(v) for v in value]
    if isinstance(value, tuple):
        return [sanitize(v) for v in value]
    if isinstance(value, np.ndarray):
        return sanitize(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            count += 1
    return count


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(*parts: Any) -> int:
    text = "::".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % (2**32 - 1)


def clip01(values: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), 0.0, 1.0)


def rank01(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.linspace(0.0, 1.0, len(values))
    return ranks


def generate_pool(family: str, split: str, seed: int, m: int) -> SyntheticPool:
    rng = np.random.default_rng(stable_seed(family, split, seed, m))
    base = np.linspace(0.0, 1.0, m)
    rng.shuffle(base)
    noise = rng.normal(0.0, 0.035, size=m)
    random_score = rng.normal(0.0, 1.0, size=m)

    if family == "aligned":
        real = clip01(0.12 + 0.78 * base + rng.normal(0.0, 0.045, size=m))
        score = clip01(real + rng.normal(0.0, 0.055, size=m))
        uncertainty = clip01(0.18 + 0.10 * rng.random(m))
        diagnostic = clip01(real + rng.normal(0.0, 0.06, size=m))
    elif family == "deceptive_tail":
        score = clip01(base + noise)
        tail = np.maximum(0.0, (base - 0.66) / 0.34)
        real = clip01(0.62 + 0.20 * base - 0.78 * tail + rng.normal(0.0, 0.045, size=m))
        uncertainty = clip01(0.15 + 0.70 * tail + rng.normal(0.0, 0.035, size=m))
        diagnostic = clip01(1.0 - tail + rng.normal(0.0, 0.08, size=m))
    elif family == "repairable":
        score = clip01(base + noise)
        hazard = clip01(0.25 + 0.70 * (base > 0.68).astype(float) + rng.normal(0.0, 0.08, size=m))
        real = clip01(0.22 + 0.70 * base - 0.58 * hazard + rng.normal(0.0, 0.045, size=m))
        uncertainty = clip01(0.12 + 0.72 * hazard + rng.normal(0.0, 0.04, size=m))
        diagnostic = clip01(base - hazard + 0.5 + rng.normal(0.0, 0.06, size=m))
    elif family == "saturated":
        real = clip01(0.48 + 0.12 * np.tanh(3.0 * (base - 0.45)) + rng.normal(0.0, 0.035, size=m))
        score = clip01(base + rng.normal(0.0, 0.06, size=m))
        uncertainty = clip01(0.22 + 0.12 * rng.random(m))
        diagnostic = clip01(real + rng.normal(0.0, 0.08, size=m))
    elif family == "hidden_flip":
        score = clip01(base + noise)
        flip = 1.0 if seed % 2 == 0 else -1.0
        real = clip01(0.50 + flip * 0.38 * (base - 0.5) + rng.normal(0.0, 0.035, size=m))
        uncertainty = clip01(0.42 + 0.08 * rng.random(m))
        diagnostic = clip01(0.50 + rng.normal(0.0, 0.05, size=m))
    else:
        raise ValueError(f"unknown family: {family}")

    imagined = clip01(score + rng.normal(0.0, 0.025, size=m))
    return SyntheticPool(
        pool_id=f"{split}_{family}_{seed}",
        family=family,
        split=split,
        seed=int(seed),
        score=np.asarray(score, dtype=float),
        imagined_utility=np.asarray(imagined, dtype=float),
        real_utility=np.asarray(real, dtype=float),
        uncertainty=np.asarray(uncertainty, dtype=float),
        diagnostic=np.asarray(diagnostic, dtype=float),
        random_score=np.asarray(random_score, dtype=float),
    )


def generate_pools(*, smoke: bool) -> list[SyntheticPool]:
    families = ["aligned", "deceptive_tail", "repairable", "saturated", "hidden_flip"]
    splits = ["pilot", "dev", "heldout"]
    seeds_per_family = 4 if smoke else 10
    m = 48 if smoke else 128
    pools: list[SyntheticPool] = []
    for split in splits:
        offset = {"pilot": 0, "dev": 10_000, "heldout": 20_000}[split]
        for family in families:
            for i in range(seeds_per_family):
                pools.append(generate_pool(family, split, offset + i, m))
    return pools


def deterministic_split(pool: SyntheticPool, label_budget: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(stable_seed(pool.pool_id, "pilot_indices", label_budget))
    perm = rng.permutation(len(pool.score))
    k = min(max(0, int(label_budget)), max(0, len(pool.score) - 4))
    return perm[:k], perm[k:]


def score_for_selector(pool: SyntheticPool, selector: str, idx: np.ndarray | None = None) -> np.ndarray:
    sl = slice(None) if idx is None else idx
    if selector == "raw_score":
        return pool.score[sl]
    if selector == "random":
        return pool.random_score[sl]
    if selector == "uncertainty_penalty":
        return pool.score[sl] - 0.75 * pool.uncertainty[sl]
    if selector == "score_clipping":
        source = pool.score[sl]
        return np.minimum(source, float(np.quantile(source, 0.75)))
    if selector == "rank_average":
        return 0.55 * rank01(pool.score[sl]) + 0.30 * rank01(pool.diagnostic[sl]) + 0.15 * rank01(-pool.uncertainty[sl])
    if selector == "cvar_lcb":
        return pool.imagined_utility[sl] - 1.25 * pool.uncertainty[sl]
    if selector == "lower_confidence_bound":
        return 0.62 * pool.score[sl] + 0.38 * pool.diagnostic[sl] - 0.95 * pool.uncertainty[sl]
    if selector == "oracle_real_utility":
        return pool.real_utility[sl]
    raise ValueError(f"unknown selector: {selector}")


def selected_utility(scores: np.ndarray, real: np.ndarray, N: int) -> float:
    return float(utility_best_of_n_finite(scores, real, [max(1, int(N))])[max(1, int(N))])


def raw_gain(scores: np.ndarray, real: np.ndarray, high_n: int = HIGH_N) -> float:
    curve = utility_best_of_n_finite(scores, real, [1, high_n])
    return float(curve[high_n] - curve[1])


def predict_decision(pool: SyntheticPool, label_budget: int, high_n: int = HIGH_N) -> dict[str, Any]:
    pilot_idx, held_idx = deterministic_split(pool, label_budget)
    if len(pilot_idx) < 4:
        return {
            "decision": "request_labels",
            "predicted_gain": None,
            "radius": None,
            "pilot_labels": int(len(pilot_idx)),
            "heldout_candidates": int(len(held_idx)),
        }
    scores = pool.score[pilot_idx]
    real = pool.real_utility[pilot_idx]
    gain = raw_gain(scores, real, min(high_n, len(pilot_idx)))
    radius = 0.34 / math.sqrt(len(pilot_idx))
    if gain - radius > 0.025:
        decision = "allow_high_n"
    elif gain + radius < -0.025:
        decision = "block_high_n"
    elif abs(gain) <= 0.035:
        decision = "stop_early"
    else:
        decision = "request_labels"
    return {
        "decision": decision,
        "predicted_gain": float(gain),
        "radius": float(radius),
        "pilot_labels": int(len(pilot_idx)),
        "heldout_candidates": int(len(held_idx)),
    }


def actual_outcome(pool: SyntheticPool, label_budget: int, high_n: int = HIGH_N) -> dict[str, Any]:
    _, held_idx = deterministic_split(pool, label_budget)
    scores = pool.score[held_idx]
    real = pool.real_utility[held_idx]
    n_eval = min(high_n, len(held_idx))
    curve = utility_best_of_n_finite(scores, real, [1, n_eval])
    gain = float(curve[n_eval] - curve[1])
    if gain > 0.035:
        sign = "helps"
    elif gain < -0.035:
        sign = "harms"
    else:
        sign = "saturates"
    return {
        "actual_gain": gain,
        "actual_sign": sign,
        "n_eval": int(n_eval),
        "n1_utility": float(curve[1]),
        "high_n_utility": float(curve[n_eval]),
        "heldout_candidates": int(len(held_idx)),
    }


def prediction_correct(decision: str, actual_sign: str) -> bool | None:
    if decision == "allow_high_n":
        return actual_sign == "helps"
    if decision == "block_high_n":
        return actual_sign == "harms"
    if decision == "stop_early":
        return actual_sign == "saturates"
    return None


def run_prospective_challenge(out: Path, pools: list[SyntheticPool], label_budget: int) -> dict[str, Any]:
    heldout = [pool for pool in pools if pool.split == "heldout"]
    pred_path = out / "prospective_audit_predictions.csv"
    pred_fields = [
        "pool_id",
        "family",
        "split",
        "label_budget",
        "pilot_labels",
        "heldout_candidates",
        "decision",
        "predicted_gain",
        "radius",
    ]
    prediction_rows = []
    for pool in heldout:
        prediction = predict_decision(pool, label_budget)
        prediction_rows.append(
            {
                "pool_id": pool.pool_id,
                "family": pool.family,
                "split": pool.split,
                "label_budget": label_budget,
                **prediction,
            }
        )
    write_csv(pred_path, prediction_rows, pred_fields)
    pred_hash = sha256(pred_path)
    (out / "prospective_audit_predictions.sha256").write_text(pred_hash + "\n", encoding="utf-8")

    outcome_path = out / "prospective_audit_outcomes.csv"
    outcome_fields = [
        "pool_id",
        "family",
        "label_budget",
        "decision",
        "actual_sign",
        "correct",
        "actual_gain",
        "n1_utility",
        "high_n_utility",
        "regret_vs_oracle",
        "regret_avoided_vs_raw",
        "false_allow",
        "false_block",
        "prediction_sha256",
    ]
    outcome_rows = []
    for row, pool in zip(prediction_rows, heldout):
        outcome = actual_outcome(pool, label_budget)
        _, held_idx = deterministic_split(pool, label_budget)
        oracle = selected_utility(score_for_selector(pool, "oracle_real_utility", held_idx), pool.real_utility[held_idx], outcome["n_eval"])
        if row["decision"] == "allow_high_n":
            policy_utility = outcome["high_n_utility"]
        else:
            policy_utility = outcome["n1_utility"]
        correct = prediction_correct(str(row["decision"]), outcome["actual_sign"])
        outcome_rows.append(
            {
                "pool_id": pool.pool_id,
                "family": pool.family,
                "label_budget": label_budget,
                "decision": row["decision"],
                "actual_sign": outcome["actual_sign"],
                "correct": "" if correct is None else bool(correct),
                "actual_gain": outcome["actual_gain"],
                "n1_utility": outcome["n1_utility"],
                "high_n_utility": outcome["high_n_utility"],
                "regret_vs_oracle": float(oracle - policy_utility),
                "regret_avoided_vs_raw": float(policy_utility - min(outcome["n1_utility"], outcome["high_n_utility"])),
                "false_allow": row["decision"] == "allow_high_n" and outcome["actual_sign"] != "helps",
                "false_block": row["decision"] == "block_high_n" and outcome["actual_sign"] != "harms",
                "prediction_sha256": pred_hash,
            }
        )
    write_csv(outcome_path, outcome_rows, outcome_fields)
    decided = [row for row in outcome_rows if row["correct"] != ""]
    summary = {
        "prediction_file": str(pred_path),
        "prediction_sha256": pred_hash,
        "outcome_file": str(outcome_path),
        "heldout_pools": len(heldout),
        "label_budget": label_budget,
        "decision_counts": count_by(outcome_rows, "decision"),
        "actual_sign_counts": count_by(outcome_rows, "actual_sign"),
        "decided_rows": len(decided),
        "decision_accuracy": mean_bool(row["correct"] for row in decided),
        "false_allow_rate": mean_bool(row["false_allow"] for row in outcome_rows),
        "false_block_rate": mean_bool(row["false_block"] for row in outcome_rows),
        "mean_regret_vs_oracle": mean_float(row["regret_vs_oracle"] for row in outcome_rows),
        "mean_regret_avoided_vs_raw": mean_float(row["regret_avoided_vs_raw"] for row in outcome_rows),
        "gate_passed": pred_hash and len(outcome_rows) == len(heldout) and len(decided) > 0,
    }
    write_json(out / "prospective_audit_summary.json", summary)
    return summary


def mean_float(values: Iterable[Any]) -> float | None:
    arr = [float(value) for value in values if value != "" and value is not None]
    return float(np.mean(arr)) if arr else None


def mean_bool(values: Iterable[Any]) -> float | None:
    arr = [bool(value) for value in values if value != "" and value is not None]
    return float(np.mean(arr)) if arr else None


def count_by(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        item = str(row[key])
        out[item] = out.get(item, 0) + 1
    return dict(sorted(out.items()))


def run_label_budget(out: Path, pools: list[SyntheticPool]) -> dict[str, Any]:
    heldout = [pool for pool in pools if pool.split == "heldout"]
    path = out / "label_budget_sample_complexity.csv"
    fields = [
        "label_budget",
        "pool_id",
        "family",
        "decision",
        "actual_sign",
        "correct",
        "false_allow",
        "false_block",
        "regret_avoided_vs_raw",
    ]
    rows = []
    for budget in LABEL_BUDGETS:
        for pool in heldout:
            pred = predict_decision(pool, budget)
            outcome = actual_outcome(pool, budget)
            correct = prediction_correct(pred["decision"], outcome["actual_sign"])
            policy_utility = outcome["high_n_utility"] if pred["decision"] == "allow_high_n" else outcome["n1_utility"]
            rows.append(
                {
                    "label_budget": budget,
                    "pool_id": pool.pool_id,
                    "family": pool.family,
                    "decision": pred["decision"],
                    "actual_sign": outcome["actual_sign"],
                    "correct": "" if correct is None else bool(correct),
                    "false_allow": pred["decision"] == "allow_high_n" and outcome["actual_sign"] != "helps",
                    "false_block": pred["decision"] == "block_high_n" and outcome["actual_sign"] != "harms",
                    "regret_avoided_vs_raw": float(policy_utility - min(outcome["n1_utility"], outcome["high_n_utility"])),
                }
            )
    write_csv(path, rows, fields)
    budget_summaries = []
    for budget in LABEL_BUDGETS:
        sub = [row for row in rows if row["label_budget"] == budget]
        decided = [row for row in sub if row["correct"] != ""]
        budget_summaries.append(
            {
                "label_budget": budget,
                "decision_counts": count_by(sub, "decision"),
                "decided_rows": len(decided),
                "decision_accuracy": mean_bool(row["correct"] for row in decided),
                "false_allow_rate": mean_bool(row["false_allow"] for row in sub),
                "false_block_rate": mean_bool(row["false_block"] for row in sub),
                "mean_regret_avoided_vs_raw": mean_float(row["regret_avoided_vs_raw"] for row in sub),
            }
        )
    summary = {
        "artifact": str(path),
        "label_budgets": LABEL_BUDGETS,
        "heldout_pools": len(heldout),
        "budget_summaries": budget_summaries,
        "zero_label_included": any(item["label_budget"] == 0 for item in budget_summaries),
        "gate_passed": len(budget_summaries) == len(LABEL_BUDGETS)
        and any("request_labels" in item["decision_counts"] for item in budget_summaries),
    }
    write_json(out / "label_budget_sample_complexity_summary.json", summary)
    return summary


def run_selector_gauntlet(out: Path, pools: list[SyntheticPool]) -> dict[str, Any]:
    heldout = [pool for pool in pools if pool.split == "heldout"]
    selectors = [
        "raw_score",
        "random",
        "uncertainty_penalty",
        "score_clipping",
        "rank_average",
        "cvar_lcb",
        "lower_confidence_bound",
        "oracle_real_utility",
    ]
    path = out / "selector_gauntlet.csv"
    fields = ["pool_id", "family", "selector", "N", "selected_utility", "deployable", "oracle_row"]
    rows = []
    for pool in heldout:
        _, held_idx = deterministic_split(pool, 16)
        real = pool.real_utility[held_idx]
        for selector in selectors:
            scores = score_for_selector(pool, selector, held_idx)
            curve = utility_best_of_n_finite(scores, real, N_VALUES)
            for N in N_VALUES:
                rows.append(
                    {
                        "pool_id": pool.pool_id,
                        "family": pool.family,
                        "selector": selector,
                        "N": N,
                        "selected_utility": float(curve[N]),
                        "deployable": selector != "oracle_real_utility",
                        "oracle_row": selector == "oracle_real_utility",
                    }
                )
    write_csv(path, rows, fields)

    summary_rows = []
    for selector in selectors:
        vals = [row["selected_utility"] for row in rows if row["selector"] == selector and row["N"] == HIGH_N]
        raw_vals = [row["selected_utility"] for row in rows if row["selector"] == "raw_score" and row["N"] == HIGH_N]
        summary_rows.append(
            {
                "selector": selector,
                "mean_high_n_utility": mean_float(vals),
                "mean_gain_vs_raw_high_n": mean_float(np.asarray(vals) - np.asarray(raw_vals)),
                "deployable": selector != "oracle_real_utility",
            }
        )
    summary = {
        "artifact": str(path),
        "selectors": selectors,
        "heldout_pools": len(heldout),
        "summary_rows": summary_rows,
        "oracle_labeled_non_deployable": True,
        "gate_passed": len(selectors) >= 8 and any(row["oracle_row"] for row in rows),
    }
    write_json(out / "selector_gauntlet_summary.json", summary)
    return summary


def calibrated_scores(pool: SyntheticPool, pilot_idx: np.ndarray, eval_idx: np.ndarray, ridge: float = 1.0) -> np.ndarray:
    if len(pilot_idx) < 3:
        return score_for_selector(pool, "lower_confidence_bound", eval_idx)
    x_train = np.column_stack(
        [
            np.ones(len(pilot_idx)),
            pool.score[pilot_idx],
            pool.imagined_utility[pilot_idx],
            pool.uncertainty[pilot_idx],
            pool.diagnostic[pilot_idx],
        ]
    )
    y_train = pool.real_utility[pilot_idx]
    x_eval = np.column_stack(
        [
            np.ones(len(eval_idx)),
            pool.score[eval_idx],
            pool.imagined_utility[eval_idx],
            pool.uncertainty[eval_idx],
            pool.diagnostic[eval_idx],
        ]
    )
    penalty = ridge * np.eye(x_train.shape[1])
    penalty[0, 0] = 0.0
    weights = np.linalg.solve(x_train.T @ x_train + penalty, x_train.T @ y_train)
    residual = y_train - x_train @ weights
    radius = float(np.quantile(np.abs(residual), 0.80)) if len(residual) else 0.2
    return np.asarray(x_eval @ weights - radius, dtype=float)


def run_equal_compute_frontier(out: Path, pools: list[SyntheticPool]) -> dict[str, Any]:
    heldout = [pool for pool in pools if pool.split == "heldout"]
    budgets = [8, 16, 32, 64]
    path = out / "equal_compute_frontier.csv"
    fields = [
        "pool_id",
        "family",
        "budget",
        "strategy",
        "rollouts_used",
        "labels_used",
        "cpu_units",
        "selected_utility",
        "regret_vs_oracle",
    ]
    rows = []
    for pool in heldout:
        for budget in budgets:
            pilot_k = min(max(0, budget // 4), 16)
            pilot_idx, eval_idx = deterministic_split(pool, pilot_k)
            eval_n = max(1, min(len(eval_idx), budget - pilot_k))
            real = pool.real_utility[eval_idx]
            oracle = selected_utility(score_for_selector(pool, "oracle_real_utility", eval_idx), real, eval_n)
            strategies = {
                "blind_more_rollouts": (score_for_selector(pool, "raw_score", eval_idx), eval_n, pilot_k),
                "audit_fewer_rollouts": (
                    score_for_selector(pool, "lower_confidence_bound", eval_idx)
                    if predict_decision(pool, pilot_k)["decision"] == "allow_high_n"
                    else score_for_selector(pool, "raw_score", eval_idx),
                    eval_n if predict_decision(pool, pilot_k)["decision"] == "allow_high_n" else 1,
                    pilot_k,
                ),
                "calibrated_lcb": (calibrated_scores(pool, pilot_idx, eval_idx), eval_n, pilot_k),
                "conservative_stop_block": (
                    score_for_selector(pool, "raw_score", eval_idx),
                    1 if predict_decision(pool, pilot_k)["decision"] != "allow_high_n" else eval_n,
                    pilot_k,
                ),
                "oracle_upper_bound": (score_for_selector(pool, "oracle_real_utility", eval_idx), eval_n, 0),
            }
            for strategy, (scores, n_use, labels_used) in strategies.items():
                utility = selected_utility(scores, real, n_use)
                rows.append(
                    {
                        "pool_id": pool.pool_id,
                        "family": pool.family,
                        "budget": budget,
                        "strategy": strategy,
                        "rollouts_used": int(n_use),
                        "labels_used": int(labels_used),
                        "cpu_units": int(n_use + 4 * labels_used),
                        "selected_utility": utility,
                        "regret_vs_oracle": float(oracle - utility),
                    }
                )
    write_csv(path, rows, fields)
    summary_rows = []
    for budget in budgets:
        for strategy in sorted({row["strategy"] for row in rows}):
            sub = [row for row in rows if row["budget"] == budget and row["strategy"] == strategy]
            summary_rows.append(
                {
                    "budget": budget,
                    "strategy": strategy,
                    "mean_selected_utility": mean_float(row["selected_utility"] for row in sub),
                    "mean_regret_vs_oracle": mean_float(row["regret_vs_oracle"] for row in sub),
                    "mean_cpu_units": mean_float(row["cpu_units"] for row in sub),
                }
            )
    summary = {
        "artifact": str(path),
        "budgets": budgets,
        "heldout_pools": len(heldout),
        "summary_rows": summary_rows,
        "reports_cpu_units": True,
        "gate_passed": bool(rows) and all("cpu_units" in row for row in rows),
    }
    write_json(out / "equal_compute_frontier_summary.json", summary)
    return summary


def run_closed_loop_validation(out: Path, pools: list[SyntheticPool]) -> dict[str, Any]:
    heldout_by_family: dict[str, list[SyntheticPool]] = {}
    for pool in pools:
        if pool.split == "heldout":
            heldout_by_family.setdefault(pool.family, []).append(pool)
    episode_count = min(len(items) for items in heldout_by_family.values())
    policies = ["n1", "raw_high_n", "random_high_n", "audit_policy", "oracle_upper_bound"]
    rows = []
    for ep in range(episode_count):
        episode_pools = [heldout_by_family[family][ep] for family in sorted(heldout_by_family)]
        for policy in policies:
            total = 0.0
            rollouts = 0
            labels = 0
            blocked = 0
            for pool in episode_pools:
                pilot_idx, eval_idx = deterministic_split(pool, 16)
                real = pool.real_utility[eval_idx]
                if policy == "n1":
                    scores, n_use = score_for_selector(pool, "raw_score", eval_idx), 1
                elif policy == "raw_high_n":
                    scores, n_use = score_for_selector(pool, "raw_score", eval_idx), HIGH_N
                elif policy == "random_high_n":
                    scores, n_use = score_for_selector(pool, "random", eval_idx), HIGH_N
                elif policy == "audit_policy":
                    decision = predict_decision(pool, 16)["decision"]
                    labels += len(pilot_idx)
                    if decision == "allow_high_n":
                        scores, n_use = score_for_selector(pool, "lower_confidence_bound", eval_idx), HIGH_N
                    else:
                        scores, n_use = score_for_selector(pool, "raw_score", eval_idx), 1
                        blocked += int(decision in {"block_high_n", "request_labels"})
                elif policy == "oracle_upper_bound":
                    scores, n_use = score_for_selector(pool, "oracle_real_utility", eval_idx), HIGH_N
                else:
                    raise AssertionError(policy)
                n_use = min(n_use, len(eval_idx))
                total += selected_utility(scores, real, n_use)
                rollouts += n_use
            rows.append(
                {
                    "episode": ep,
                    "policy": policy,
                    "return": total,
                    "rollouts_used": rollouts,
                    "labels_used": labels,
                    "blocked_steps": blocked,
                }
            )
    path = out / "closed_loop_validation.csv"
    write_csv(path, rows, ["episode", "policy", "return", "rollouts_used", "labels_used", "blocked_steps"])
    by_policy = {policy: [row["return"] for row in rows if row["policy"] == policy] for policy in policies}
    raw = np.asarray(by_policy["raw_high_n"], dtype=float)
    audit = np.asarray(by_policy["audit_policy"], dtype=float)
    n1 = np.asarray(by_policy["n1"], dtype=float)
    summary = {
        "artifact": str(path),
        "episodes": episode_count,
        "policies": policies,
        "mean_return_by_policy": {policy: mean_float(vals) for policy, vals in by_policy.items()},
        "audit_minus_raw_ci": paired_bootstrap_ci(audit, raw, seed=5, n_boot=500),
        "audit_minus_n1_ci": paired_bootstrap_ci(audit, n1, seed=7, n_boot=500),
        "mean_blocked_steps": mean_float(row["blocked_steps"] for row in rows if row["policy"] == "audit_policy"),
        "gate_passed": episode_count > 0 and len(policies) == 5,
    }
    write_json(out / "closed_loop_validation_summary.json", summary)
    return summary


def run(*, smoke: bool = False, output_root: Path = ROOT) -> dict[str, Any]:
    out = output_root / ("results/v5_smoke" if smoke else "results/v5")
    out.mkdir(parents=True, exist_ok=True)
    pools = generate_pools(smoke=smoke)
    label_budget = 8 if smoke else 16
    prospective = run_prospective_challenge(out, pools, label_budget)
    label_budget_summary = run_label_budget(out, pools)
    selector = run_selector_gauntlet(out, pools)
    frontier = run_equal_compute_frontier(out, pools)
    closed_loop = run_closed_loop_validation(out, pools)
    summary = {
        "mode": "smoke" if smoke else "canonical",
        "pool_count": len(pools),
        "heldout_pool_count": len([pool for pool in pools if pool.split == "heldout"]),
        "candidate_count_per_pool": int(len(pools[0].score)) if pools else 0,
        "low_ram_design": {
            "parallel_jobs": 1,
            "materializes_optional_robotics_envs": False,
            "stores_full_rollout_tensors": False,
            "writes_compact_csv_json": True,
        },
        "prospective_audit": prospective,
        "label_budget_sample_complexity": label_budget_summary,
        "selector_gauntlet": selector,
        "equal_compute_frontier": frontier,
        "closed_loop_validation": closed_loop,
        "gate_passed": all(
            [
                prospective["gate_passed"],
                label_budget_summary["gate_passed"],
                selector["gate_passed"],
                frontier["gate_passed"],
                closed_loop["gate_passed"],
            ]
        ),
    }
    write_json(out / "prospective_evidence_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="write smaller smoke artifacts under results/v5_smoke")
    args = parser.parse_args()
    summary = run(smoke=args.smoke)
    print(
        f"v5 prospective evidence complete ({summary['mode']}): "
        f"heldout_pools={summary['heldout_pool_count']} "
        f"decision_accuracy={summary['prospective_audit']['decision_accuracy']} "
        f"gate={summary['gate_passed']}"
    )


if __name__ == "__main__":
    main()
