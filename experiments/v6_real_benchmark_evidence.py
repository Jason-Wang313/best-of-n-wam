from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "v6"


SOURCE_SPECS: list[dict[str, Any]] = [
    {
        "family": "gym_reacher",
        "path": "benchmark_gym_manip_curves.csv",
        "keys": ["seed", "state_id"],
        "target_priority": ["learned_predicted_utility", "low_energy"],
    },
    {
        "family": "fetch_state",
        "path": "benchmark_gym_robotics_curves.csv",
        "keys": ["benchmark", "seed", "state_id"],
        "target_priority": ["learned_wam", "predicted_success", "low_energy"],
    },
    {
        "family": "fetch_visual",
        "path": "benchmark_gym_robotics_visual_wam_curves.csv",
        "keys": ["benchmark", "seed", "state_id"],
        "target_priority": ["visual_wam", "visual_success_head", "low_energy"],
    },
    {
        "family": "maniskill",
        "path": "benchmark_maniskill_curves.csv",
        "keys": ["benchmark", "seed", "state_id"],
        "target_priority": ["dense_reward", "learned_predicted_utility", "low_energy"],
    },
    {
        "family": "metaworld",
        "path": "benchmark_metaworld_curves.csv",
        "keys": ["benchmark", "seed", "state_id"],
        "target_priority": ["learned_wam", "benchmark_reward", "predicted_success", "low_energy"],
    },
    {
        "family": "robosuite",
        "path": "benchmark_robosuite_curves.csv",
        "keys": ["benchmark", "seed", "state_id"],
        "target_priority": ["learned_wam", "benchmark_reward", "progress", "predicted_success", "low_energy"],
    },
    {
        "family": "libero",
        "path": "benchmark_libero_curves.csv",
        "keys": ["task_key", "seed", "state_id"],
        "target_priority": [
            "learned_energy_regularized",
            "learned_wam",
            "learned_physics_score",
            "distance_progress",
            "low_energy",
        ],
    },
    {
        "family": "robocasa_family32",
        "path": "benchmark_robocasa_family32_curves.csv",
        "keys": ["env_id", "seed", "state_id"],
        "target_priority": [
            "learned_energy_regularized",
            "learned_wam",
            "learned_physics_score",
            "distance_progress",
            "low_energy",
        ],
    },
    {
        "family": "robocasa_stratified97",
        "path": "benchmark_robocasa_stratified97_curves.csv",
        "keys": ["env_id", "seed", "state_id"],
        "target_priority": [
            "learned_wam",
            "learned_energy_regularized",
            "learned_physics_score",
            "distance_progress",
            "low_energy",
        ],
    },
    {
        "family": "robocasa_residual35",
        "path": "benchmark_robocasa_residual35_h1_n4_curves.csv",
        "keys": ["env_id", "seed", "state_id"],
        "target_priority": [
            "learned_wam",
            "learned_energy_regularized",
            "learned_physics_score",
            "distance_progress",
            "low_energy",
        ],
    },
]


PRIMARY_EPSILON = 0.02
PRIMARY_RADIUS_QUANTILE = 0.90
ROBUSTNESS_EPSILONS = [0.0, 0.01, 0.02, 0.05]
ROBUSTNESS_QUANTILES = [0.50, 0.75, 0.90]
ROBUSTNESS_PILOTS = [2, 4]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    pos = (len(xs) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def label_for_gain(gain: float, epsilon: float) -> str:
    if gain > epsilon:
        return "helps"
    if gain < -epsilon:
        return "harms"
    return "saturates"


def decision_from_interval(predicted_gain: float, radius: float, epsilon: float) -> tuple[str, float, float]:
    lcb = predicted_gain - radius
    ucb = predicted_gain + radius
    if lcb > epsilon:
        return "allow_high_n", lcb, ucb
    if ucb < -epsilon:
        return "block_high_n", lcb, ucb
    if abs(predicted_gain) <= epsilon:
        return "stop_early", lcb, ucb
    return "request_labels", lcb, ucb


def decision_correct(decision: str, actual_gain: float, epsilon: float) -> bool | None:
    if decision == "request_labels":
        return None
    actual = label_for_gain(actual_gain, epsilon)
    if decision == "allow_high_n":
        return actual == "helps"
    if decision == "block_high_n":
        return actual == "harms"
    if decision == "stop_early":
        return actual == "saturates"
    return False


def utility_for_decision(record: dict[str, Any], decision: str) -> float:
    return float(record["high_utility"] if decision == "allow_high_n" else record["n1_utility"])


def rollout_units_for_decision(record: dict[str, Any], decision: str) -> int:
    return int(record["high_n"] if decision == "allow_high_n" else record["pilot_n"])


def metric_value(row: dict[str, str], metric_col: str, min_value: float, scale: float) -> float:
    value = float(row[metric_col])
    if metric_col == "normalized_real_utility":
        return value
    return (value - min_value) / scale


def load_family_records(source_root: Path, spec: dict[str, Any], max_pools: int | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = source_root / "results" / "tables" / spec["path"]
    if not path.exists():
        return [], {"family": spec["family"], "path": str(path), "exists": False}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        raw_rows = list(reader)
        fieldnames = reader.fieldnames or []
    if not raw_rows or "scorer" not in fieldnames or "N" not in fieldnames:
        return [], {"family": spec["family"], "path": str(path), "exists": True, "usable": False}

    metric_col = "normalized_real_utility" if "normalized_real_utility" in fieldnames else "real_utility"
    raw_values = [float(row[metric_col]) for row in raw_rows]
    min_value = min(raw_values)
    max_value = max(raw_values)
    scale = max(max_value - min_value, 1e-12)

    groups: dict[tuple[str, ...], dict[str, dict[int, float]]] = {}
    labels: dict[tuple[str, ...], str] = {}
    for row in raw_rows:
        key = tuple(row.get(col, "") for col in spec["keys"])
        score = row["scorer"]
        n = int(float(row["N"]))
        groups.setdefault(key, {}).setdefault(score, {})[n] = metric_value(row, metric_col, min_value, scale)
        labels[key] = row.get("benchmark") or row.get("env_id") or row.get("task_key") or spec["family"]

    records: list[dict[str, Any]] = []
    for key, scorer_map in groups.items():
        scorers = set(scorer_map)
        target = next((name for name in spec["target_priority"] if name in scorers), None)
        if not target or "random" not in scorer_map or "oracle_real_utility" not in scorer_map:
            continue
        target_ns = sorted(scorer_map[target])
        if 1 not in target_ns:
            continue
        pilot_n = 4 if 4 in target_ns else 2 if 2 in target_ns else None
        if pilot_n is None:
            continue
        high_n = max(target_ns)
        if high_n not in scorer_map["random"] or high_n not in scorer_map["oracle_real_utility"]:
            continue
        nonoracle_high = [
            values[high_n]
            for scorer, values in scorer_map.items()
            if scorer not in {"oracle_real_utility", "random"} and high_n in values
        ]
        if not nonoracle_high:
            continue
        pool_id = f"{spec['family']}:" + "|".join(key)
        records.append(
            {
                "pool_id": pool_id,
                "family": spec["family"],
                "source_file": spec["path"],
                "source_sha256": sha256(path),
                "task_label": labels.get(key, spec["family"]),
                "key": "|".join(key),
                "target_scorer": target,
                "pilot_n": pilot_n,
                "high_n": high_n,
                "n1_utility": scorer_map[target][1],
                "pilot_utility": scorer_map[target][pilot_n],
                "high_utility": scorer_map[target][high_n],
                "random_high_utility": scorer_map["random"][high_n],
                "oracle_high_utility": scorer_map["oracle_real_utility"][high_n],
                "best_nonoracle_high_utility": max(nonoracle_high),
                "worst_nonoracle_high_utility": min(nonoracle_high),
            }
        )
        if max_pools is not None and len(records) >= max_pools:
            break

    return records, {
        "family": spec["family"],
        "path": str(path),
        "relative_path": f"results/tables/{spec['path']}",
        "exists": True,
        "usable": bool(records),
        "rows": len(raw_rows),
        "pools": len(records),
        "metric_col": metric_col,
        "sha256": sha256(path),
        "target_priority": spec["target_priority"],
    }


def load_records(source_root: Path, *, smoke: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    specs = SOURCE_SPECS
    all_records: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    max_pools = 8 if smoke else None
    for spec in specs:
        records, source = load_family_records(source_root, spec, max_pools=max_pools)
        sources.append(source)
        all_records.extend(records)
    return all_records, sources


def fold_params(train: list[dict[str, Any]], *, pilot_n: int | None, radius_quantile: float) -> dict[str, Any]:
    residuals: list[float] = []
    for record in train:
        if pilot_n is not None and int(record["pilot_n"]) != pilot_n:
            continue
        residuals.append(
            (float(record["high_utility"]) - float(record["n1_utility"]))
            - (float(record["pilot_utility"]) - float(record["n1_utility"]))
        )
    if not residuals:
        residuals = [
            (float(record["high_utility"]) - float(record["n1_utility"]))
            - (float(record["pilot_utility"]) - float(record["n1_utility"]))
            for record in train
        ]
    median_residual = statistics.median(residuals) if residuals else 0.0
    abs_centered = [abs(value - median_residual) for value in residuals]
    radius = quantile(abs_centered, radius_quantile)
    train_coverage = mean([1.0 if abs(value - median_residual) <= radius + 1e-12 else 0.0 for value in residuals])
    return {
        "median_residual": median_residual,
        "radius": radius,
        "radius_quantile": radius_quantile,
        "train_residual_count": len(residuals),
        "train_interval_coverage": train_coverage,
    }


def predict_records(
    records: list[dict[str, Any]],
    *,
    epsilon: float = PRIMARY_EPSILON,
    radius_quantile: float = PRIMARY_RADIUS_QUANTILE,
    pilot_n: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    families = sorted({str(record["family"]) for record in records})
    predictions: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    for holdout in families:
        train = [record for record in records if record["family"] != holdout]
        test = [record for record in records if record["family"] == holdout]
        params = fold_params(train, pilot_n=pilot_n, radius_quantile=radius_quantile)
        calibration_rows.append(
            {
                "holdout_family": holdout,
                "train_families": ";".join(family for family in families if family != holdout),
                "train_pool_count": len(train),
                **params,
            }
        )
        for record in test:
            if pilot_n is not None and int(record["pilot_n"]) != pilot_n:
                continue
            pilot_gain = float(record["pilot_utility"]) - float(record["n1_utility"])
            predicted_gain = pilot_gain + float(params["median_residual"])
            decision, lcb, ucb = decision_from_interval(predicted_gain, float(params["radius"]), epsilon)
            predictions.append(
                {
                    "pool_id": record["pool_id"],
                    "family": record["family"],
                    "task_label": record["task_label"],
                    "key": record["key"],
                    "target_scorer": record["target_scorer"],
                    "pilot_n": record["pilot_n"],
                    "high_n": record["high_n"],
                    "epsilon": epsilon,
                    "radius_quantile": radius_quantile,
                    "pilot_gain": pilot_gain,
                    "predicted_gain": predicted_gain,
                    "lcb": lcb,
                    "ucb": ucb,
                    "decision": decision,
                    "holdout_family": holdout,
                    "train_residual_count": params["train_residual_count"],
                    "calibration_radius": params["radius"],
                    "calibration_median_residual": params["median_residual"],
                }
            )
    return predictions, calibration_rows


def attach_outcomes(predictions: list[dict[str, Any]], records: list[dict[str, Any]], prediction_sha: str) -> list[dict[str, Any]]:
    by_id = {record["pool_id"]: record for record in records}
    outcomes: list[dict[str, Any]] = []
    for pred in predictions:
        record = by_id[pred["pool_id"]]
        actual_gain = float(record["high_utility"]) - float(record["n1_utility"])
        correct = decision_correct(str(pred["decision"]), actual_gain, float(pred["epsilon"]))
        audit_utility = utility_for_decision(record, str(pred["decision"]))
        audit_rollout_units = rollout_units_for_decision(record, str(pred["decision"]))
        outcomes.append(
            {
                **pred,
                "prediction_sha256": prediction_sha,
                "n1_utility": record["n1_utility"],
                "pilot_utility": record["pilot_utility"],
                "high_utility": record["high_utility"],
                "random_high_utility": record["random_high_utility"],
                "oracle_high_utility": record["oracle_high_utility"],
                "best_nonoracle_high_utility": record["best_nonoracle_high_utility"],
                "worst_nonoracle_high_utility": record["worst_nonoracle_high_utility"],
                "actual_gain": actual_gain,
                "actual_label": label_for_gain(actual_gain, float(pred["epsilon"])),
                "decision_correct": "" if correct is None else bool(correct),
                "audit_policy_utility": audit_utility,
                "audit_minus_raw_high_n": audit_utility - float(record["high_utility"]),
                "audit_rollout_units": audit_rollout_units,
                "audit_label_units": record["pilot_n"],
            }
        )
    return outcomes


def summarize_outcomes(outcomes: list[dict[str, Any]]) -> dict[str, Any]:
    decided = [row for row in outcomes if row["decision"] != "request_labels"]
    correct = [row for row in decided if row["decision_correct"] is True]
    total = max(1, len(outcomes))
    decision_counts: dict[str, int] = {}
    actual_counts: dict[str, int] = {}
    for row in outcomes:
        decision_counts[str(row["decision"])] = decision_counts.get(str(row["decision"]), 0) + 1
        actual_counts[str(row["actual_label"])] = actual_counts.get(str(row["actual_label"]), 0) + 1
    false_allow = [
        row
        for row in outcomes
        if row["decision"] == "allow_high_n" and row["actual_label"] in {"harms", "saturates"}
    ]
    false_block = [
        row
        for row in outcomes
        if row["decision"] == "block_high_n" and row["actual_label"] in {"helps", "saturates"}
    ]
    coverage = mean([1.0 if float(row["lcb"]) <= float(row["actual_gain"]) <= float(row["ucb"]) else 0.0 for row in outcomes])
    families = sorted({str(row["family"]) for row in outcomes})
    return {
        "pool_count": len(outcomes),
        "family_count": len(families),
        "families": families,
        "decision_counts": decision_counts,
        "actual_counts": actual_counts,
        "decided_pool_count": len(decided),
        "decided_rate": len(decided) / total,
        "decision_accuracy": len(correct) / len(decided) if decided else None,
        "false_allow_rate": len(false_allow) / total,
        "false_block_rate": len(false_block) / total,
        "interval_coverage": coverage,
        "mean_n1_utility": mean([float(row["n1_utility"]) for row in outcomes]),
        "mean_raw_high_n_utility": mean([float(row["high_utility"]) for row in outcomes]),
        "mean_random_high_n_utility": mean([float(row["random_high_utility"]) for row in outcomes]),
        "mean_oracle_high_n_utility": mean([float(row["oracle_high_utility"]) for row in outcomes]),
        "mean_audit_policy_utility": mean([float(row["audit_policy_utility"]) for row in outcomes]),
        "mean_audit_minus_raw_high_n": mean([float(row["audit_minus_raw_high_n"]) for row in outcomes]),
        "mean_audit_rollout_units": mean([float(row["audit_rollout_units"]) for row in outcomes]),
        "mean_audit_label_units": mean([float(row["audit_label_units"]) for row in outcomes]),
        "mean_raw_high_n_rollout_units": mean([float(row["high_n"]) for row in outcomes]),
    }


def summarize_by_family(outcomes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family in sorted({str(row["family"]) for row in outcomes}):
        subset = [row for row in outcomes if row["family"] == family]
        summary = summarize_outcomes(subset)
        rows.append(
            {
                "family": family,
                "pool_count": summary["pool_count"],
                "decided_rate": summary["decided_rate"],
                "decision_accuracy": summary["decision_accuracy"],
                "false_allow_rate": summary["false_allow_rate"],
                "false_block_rate": summary["false_block_rate"],
                "mean_audit_policy_utility": summary["mean_audit_policy_utility"],
                "mean_raw_high_n_utility": summary["mean_raw_high_n_utility"],
                "mean_audit_minus_raw_high_n": summary["mean_audit_minus_raw_high_n"],
            }
        )
    return rows


def selector_ablation(outcomes: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    strategies = {
        "n1": lambda row: (float(row["n1_utility"]), 1.0, 0.0, True),
        "raw_high_n": lambda row: (float(row["high_utility"]), float(row["high_n"]), 0.0, True),
        "random_high_n": lambda row: (float(row["random_high_utility"]), float(row["high_n"]), 0.0, True),
        "audit_with_abstention": lambda row: (
            float(row["audit_policy_utility"]),
            float(row["audit_rollout_units"]),
            float(row["audit_label_units"]),
            True,
        ),
        "pilot_sign_no_abstention": lambda row: (
            float(row["high_utility"]) if float(row["pilot_gain"]) > float(row["epsilon"]) else float(row["n1_utility"]),
            float(row["high_n"]) if float(row["pilot_gain"]) > float(row["epsilon"]) else float(row["pilot_n"]),
            float(row["pilot_n"]),
            True,
        ),
        "calibrated_sign_no_abstention": lambda row: (
            float(row["high_utility"]) if float(row["predicted_gain"]) > float(row["epsilon"]) else float(row["n1_utility"]),
            float(row["high_n"]) if float(row["predicted_gain"]) > float(row["epsilon"]) else float(row["pilot_n"]),
            float(row["pilot_n"]),
            True,
        ),
        "best_nonoracle_diagnostic": lambda row: (
            float(row["best_nonoracle_high_utility"]),
            float(row["high_n"]),
            0.0,
            False,
        ),
        "oracle_upper_bound": lambda row: (
            float(row["oracle_high_utility"]),
            float(row["high_n"]),
            0.0,
            False,
        ),
    }
    rows: list[dict[str, Any]] = []
    oracle = [float(row["oracle_high_utility"]) for row in outcomes]
    for name, fn in strategies.items():
        utilities: list[float] = []
        rollout_units: list[float] = []
        label_units: list[float] = []
        deployable = True
        for row in outcomes:
            utility, rollouts, labels, is_deployable = fn(row)
            utilities.append(utility)
            rollout_units.append(rollouts)
            label_units.append(labels)
            deployable = deployable and is_deployable
        rows.append(
            {
                "strategy": name,
                "deployable": deployable,
                "mean_utility": mean(utilities),
                "mean_rollout_units": mean(rollout_units),
                "mean_label_units": mean(label_units),
                "utility_per_rollout_unit": mean(utilities) / max(mean(rollout_units), 1e-12),
                "mean_regret_vs_oracle": mean([o - u for o, u in zip(oracle, utilities)]),
            }
        )
    by_name = {row["strategy"]: row for row in rows}
    return rows, {
        "strategy_count": len(rows),
        "deployable_strategy_count": sum(1 for row in rows if row["deployable"]),
        "audit_minus_raw_utility": by_name["audit_with_abstention"]["mean_utility"] - by_name["raw_high_n"]["mean_utility"],
        "audit_rollout_savings_vs_raw": by_name["raw_high_n"]["mean_rollout_units"] - by_name["audit_with_abstention"]["mean_rollout_units"],
        "audit_utility_per_rollout": by_name["audit_with_abstention"]["utility_per_rollout_unit"],
        "raw_utility_per_rollout": by_name["raw_high_n"]["utility_per_rollout_unit"],
    }


def robustness_grid(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pilot_n in ROBUSTNESS_PILOTS:
        eligible = [record for record in records if int(record["pilot_n"]) == pilot_n]
        if len({record["family"] for record in eligible}) < 2:
            continue
        for epsilon in ROBUSTNESS_EPSILONS:
            for q in ROBUSTNESS_QUANTILES:
                predictions, _ = predict_records(eligible, epsilon=epsilon, radius_quantile=q, pilot_n=pilot_n)
                outcomes = attach_outcomes(predictions, eligible, prediction_sha="robustness_grid")
                summary = summarize_outcomes(outcomes)
                rows.append(
                    {
                        "pilot_n": pilot_n,
                        "epsilon": epsilon,
                        "radius_quantile": q,
                        "pool_count": summary["pool_count"],
                        "decided_rate": summary["decided_rate"],
                        "decision_accuracy": summary["decision_accuracy"],
                        "false_allow_rate": summary["false_allow_rate"],
                        "false_block_rate": summary["false_block_rate"],
                        "interval_coverage": summary["interval_coverage"],
                        "mean_audit_minus_raw_high_n": summary["mean_audit_minus_raw_high_n"],
                    }
                )
    primary_like = [
        row
        for row in rows
        if row["epsilon"] == PRIMARY_EPSILON and row["radius_quantile"] == PRIMARY_RADIUS_QUANTILE
    ]
    return rows, {
        "row_count": len(rows),
        "pilot_values": sorted({row["pilot_n"] for row in rows}),
        "zero_false_allow_rows": sum(1 for row in rows if row["false_allow_rate"] == 0.0),
        "primary_like_rows": len(primary_like),
        "max_decision_accuracy": max([float(row["decision_accuracy"] or 0.0) for row in rows], default=0.0),
    }


def calibration_abstention(outcomes: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    bins = [
        ("0-0.5", 0.0, 0.5),
        ("0.5-1", 0.5, 1.0),
        ("1-2", 1.0, 2.0),
        ("2+", 2.0, math.inf),
    ]
    rows: list[dict[str, Any]] = []
    for label, lo, hi in bins:
        subset = []
        for row in outcomes:
            radius = max(abs(float(row["calibration_radius"])), 1e-12)
            margin = abs(float(row["predicted_gain"])) / radius
            if lo <= margin < hi:
                subset.append(row)
        if not subset:
            continue
        summary = summarize_outcomes(subset)
        rows.append(
            {
                "confidence_bin": label,
                "pool_count": len(subset),
                "interval_coverage": summary["interval_coverage"],
                "decided_rate": summary["decided_rate"],
                "decision_accuracy": summary["decision_accuracy"],
                "request_label_rate": summary["decision_counts"].get("request_labels", 0) / max(1, len(subset)),
                "false_allow_rate": summary["false_allow_rate"],
                "false_block_rate": summary["false_block_rate"],
            }
        )
    return rows, {
        "bin_count": len(rows),
        "overall_interval_coverage": summarize_outcomes(outcomes)["interval_coverage"],
        "overall_decided_rate": summarize_outcomes(outcomes)["decided_rate"],
        "overall_request_label_rate": summarize_outcomes(outcomes)["decision_counts"].get("request_labels", 0) / max(1, len(outcomes)),
    }


def negative_controls(outcomes: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sorted_rows = sorted(outcomes, key=lambda row: row["pool_id"])
    shifted_decisions = [row["decision"] for row in sorted_rows[-1:]] + [row["decision"] for row in sorted_rows[:-1]]
    shuffled_correct = []
    shuffled_utilities = []
    anti_utilities = []
    target_utilities = []
    for row, shifted in zip(sorted_rows, shifted_decisions):
        actual_gain = float(row["actual_gain"])
        correct = decision_correct(str(shifted), actual_gain, float(row["epsilon"]))
        if correct is not None:
            shuffled_correct.append(1.0 if correct else 0.0)
        shuffled_utilities.append(float(row["high_utility"]) if shifted == "allow_high_n" else float(row["n1_utility"]))
        anti_utilities.append(float(row["worst_nonoracle_high_utility"]))
        target_utilities.append(float(row["high_utility"]))
    rows = [
        {
            "control": "deterministic_shifted_decisions",
            "mean_accuracy_on_decided": mean(shuffled_correct),
            "mean_utility": mean(shuffled_utilities),
            "mean_gap_vs_raw_target": mean([u - raw for u, raw in zip(shuffled_utilities, target_utilities)]),
        },
        {
            "control": "worst_nonoracle_tail",
            "mean_accuracy_on_decided": "",
            "mean_utility": mean(anti_utilities),
            "mean_gap_vs_raw_target": mean([u - raw for u, raw in zip(anti_utilities, target_utilities)]),
        },
    ]
    return rows, {
        "control_count": len(rows),
        "shifted_decision_accuracy": rows[0]["mean_accuracy_on_decided"],
        "worst_nonoracle_gap_vs_raw": rows[1]["mean_gap_vs_raw_target"],
        "gate_passed": rows[1]["mean_gap_vs_raw_target"] < 0.0,
    }


def finite_sample_theory() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for epsilon in [0.02, 0.05, 0.10]:
        for delta in [0.10, 0.05, 0.01]:
            # Paired selected-utility gain is in [-1, 1] after per-family normalization.
            labels = math.ceil(2.0 * math.log(2.0 / delta) / (epsilon**2))
            rows.append(
                {
                    "epsilon": epsilon,
                    "delta": delta,
                    "paired_labels_required": labels,
                    "bound": "Hoeffding paired bounded-difference, gain in [-1,1]",
                }
            )
    eps05_delta05 = [row for row in rows if row["epsilon"] == 0.05 and row["delta"] == 0.05][0]
    return rows, {
        "row_count": len(rows),
        "epsilon_0_05_delta_0_05_labels": eps05_delta05["paired_labels_required"],
        "interpretation": "finite-sample labels are expensive, so abstention/request-label decisions are first-class outcomes",
    }


def split_manifest(records: list[dict[str, Any]], sources: list[dict[str, Any]]) -> dict[str, Any]:
    families = sorted({record["family"] for record in records})
    return {
        "protocol": "V6 leave-one-family-out real-benchmark curve audit",
        "families": families,
        "folds": [
            {
                "holdout_family": family,
                "train_families": [other for other in families if other != family],
            }
            for family in families
        ],
        "params": {
            "epsilon": PRIMARY_EPSILON,
            "radius_quantile": PRIMARY_RADIUS_QUANTILE,
            "pilot_rule": "use N=4 when present, otherwise N=2",
            "utility_normalization": "use normalized_real_utility when present; otherwise min-max scale real_utility within source CSV",
        },
        "input_artifacts": sources,
        "claims_not_supported": [
            "real robot validation",
            "GPU-scale training",
            "broad robotics SOTA",
            "universal WAM training law",
            "candidate-level replacement for unavailable simulator reruns",
        ],
    }


def run(*, output_root: Path = ROOT, source_root: Path | None = None, smoke: bool = False) -> dict[str, Any]:
    global ROOT, RESULTS, OUT
    ROOT = output_root
    RESULTS = ROOT / "results"
    OUT = RESULTS / ("v6_smoke" if smoke else "v6")
    source_root = source_root or output_root

    records, sources = load_records(source_root, smoke=smoke)
    if len(records) < (12 if smoke else 100):
        raise RuntimeError(f"not enough usable V6 real-benchmark records: {len(records)}")
    OUT.mkdir(parents=True, exist_ok=True)

    manifest = split_manifest(records, sources)
    write_json(OUT / "split_manifest.json", manifest)
    manifest_sha = sha256(OUT / "split_manifest.json")
    predictions, calibration_rows = predict_records(records)
    prediction_fields = [
        "pool_id",
        "family",
        "task_label",
        "key",
        "target_scorer",
        "pilot_n",
        "high_n",
        "epsilon",
        "radius_quantile",
        "pilot_gain",
        "predicted_gain",
        "lcb",
        "ucb",
        "decision",
        "holdout_family",
        "train_residual_count",
        "calibration_radius",
        "calibration_median_residual",
    ]
    write_csv(OUT / "real_benchmark_audit_predictions.csv", predictions, prediction_fields)
    prediction_sha = sha256(OUT / "real_benchmark_audit_predictions.csv")
    (OUT / "real_benchmark_audit_predictions.sha256").write_text(prediction_sha + "\n", encoding="utf-8")

    outcomes = attach_outcomes(predictions, records, prediction_sha)
    outcome_fields = prediction_fields + [
        "prediction_sha256",
        "n1_utility",
        "pilot_utility",
        "high_utility",
        "random_high_utility",
        "oracle_high_utility",
        "best_nonoracle_high_utility",
        "worst_nonoracle_high_utility",
        "actual_gain",
        "actual_label",
        "decision_correct",
        "audit_policy_utility",
        "audit_minus_raw_high_n",
        "audit_rollout_units",
        "audit_label_units",
    ]
    write_csv(OUT / "real_benchmark_audit_outcomes.csv", outcomes, outcome_fields)
    write_csv(
        OUT / "cross_family_calibration.csv",
        calibration_rows,
        [
            "holdout_family",
            "train_families",
            "train_pool_count",
            "median_residual",
            "radius",
            "radius_quantile",
            "train_residual_count",
            "train_interval_coverage",
        ],
    )

    family_rows = summarize_by_family(outcomes)
    write_csv(
        OUT / "cross_family_transfer.csv",
        family_rows,
        [
            "family",
            "pool_count",
            "decided_rate",
            "decision_accuracy",
            "false_allow_rate",
            "false_block_rate",
            "mean_audit_policy_utility",
            "mean_raw_high_n_utility",
            "mean_audit_minus_raw_high_n",
        ],
    )
    audit_summary = summarize_outcomes(outcomes)

    calibration_bins, calibration_summary = calibration_abstention(outcomes)
    write_csv(
        OUT / "calibration_abstention.csv",
        calibration_bins,
        [
            "confidence_bin",
            "pool_count",
            "interval_coverage",
            "decided_rate",
            "decision_accuracy",
            "request_label_rate",
            "false_allow_rate",
            "false_block_rate",
        ],
    )
    ablation_rows, ablation_summary = selector_ablation(outcomes)
    write_csv(
        OUT / "selector_metric_ablation.csv",
        ablation_rows,
        [
            "strategy",
            "deployable",
            "mean_utility",
            "mean_rollout_units",
            "mean_label_units",
            "utility_per_rollout_unit",
            "mean_regret_vs_oracle",
        ],
    )
    robustness_rows, robustness_summary = robustness_grid(records)
    write_csv(
        OUT / "robustness_grid.csv",
        robustness_rows,
        [
            "pilot_n",
            "epsilon",
            "radius_quantile",
            "pool_count",
            "decided_rate",
            "decision_accuracy",
            "false_allow_rate",
            "false_block_rate",
            "interval_coverage",
            "mean_audit_minus_raw_high_n",
        ],
    )
    negative_rows, negative_summary = negative_controls(outcomes)
    write_csv(
        OUT / "real_benchmark_negative_controls.csv",
        negative_rows,
        ["control", "mean_accuracy_on_decided", "mean_utility", "mean_gap_vs_raw_target"],
    )
    theory_rows, theory_summary = finite_sample_theory()
    write_csv(
        OUT / "finite_sample_audit_theory.csv",
        theory_rows,
        ["epsilon", "delta", "paired_labels_required", "bound"],
    )

    gate_passed = (
        audit_summary["family_count"] >= (3 if smoke else 8)
        and audit_summary["pool_count"] >= (12 if smoke else 500)
        and (audit_summary["decision_accuracy"] or 0.0) >= 0.80
        and audit_summary["false_allow_rate"] <= 0.02
        and audit_summary["false_block_rate"] <= 0.02
        and audit_summary["decided_rate"] >= 0.50
        and calibration_summary["overall_interval_coverage"] >= 0.80
        and robustness_summary["row_count"] >= (3 if smoke else 12)
        and negative_summary["gate_passed"] is True
    )
    summary = {
        "mode": "smoke" if smoke else "canonical",
        "gate_passed": bool(gate_passed),
        "low_ram_design": {
            "uses_existing_curve_csvs": True,
            "reruns_simulators": False,
            "stores_candidate_tensors": False,
            "parallel_jobs": 1,
            "compact_per_pool_records": len(records),
        },
        "source_artifacts": sources,
        "leakage_protocol": {
            "split_manifest": str(OUT / "split_manifest.json"),
            "split_manifest_sha256": manifest_sha,
            "prediction_file": str(OUT / "real_benchmark_audit_predictions.csv"),
            "prediction_sha256": prediction_sha,
            "outcome_file": str(OUT / "real_benchmark_audit_outcomes.csv"),
            "hash_locked_before_outcome_merge": True,
        },
        "real_benchmark_audit": audit_summary,
        "cross_family_transfer": {
            "artifact": str(OUT / "cross_family_transfer.csv"),
            "fold_count": len(family_rows),
            "family_rows": family_rows,
        },
        "calibration_abstention": {
            "artifact": str(OUT / "calibration_abstention.csv"),
            **calibration_summary,
        },
        "selector_metric_ablation": {
            "artifact": str(OUT / "selector_metric_ablation.csv"),
            **ablation_summary,
            "rows": ablation_rows,
        },
        "robustness_grid": {
            "artifact": str(OUT / "robustness_grid.csv"),
            **robustness_summary,
        },
        "negative_controls": {
            "artifact": str(OUT / "real_benchmark_negative_controls.csv"),
            **negative_summary,
        },
        "finite_sample_theory": {
            "artifact": str(OUT / "finite_sample_audit_theory.csv"),
            **theory_summary,
        },
    }
    write_json(OUT / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    summary = run(smoke=args.smoke)
    audit = summary["real_benchmark_audit"]
    print(
        f"v6 real-benchmark evidence complete ({summary['mode']}): "
        f"families={audit['family_count']} pools={audit['pool_count']} "
        f"accuracy={audit['decision_accuracy']:.3f} "
        f"false_allow={audit['false_allow_rate']:.3f} gate={summary['gate_passed']}"
    )


if __name__ == "__main__":
    main()
