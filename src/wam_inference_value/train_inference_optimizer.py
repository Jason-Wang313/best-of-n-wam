from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from wam_inference_value.stats import bootstrap_ci


TOY_HORIZON_BY_ENV = {
    "block_push": 12,
    "drawer_pull": 10,
    "slippery_grasp": 10,
    "nonstationary_shift": 10,
    "deformable_toy": 10,
}

LEARNED_SCORERS = {
    "predicted_utility",
    "safety_penalized",
    "uncertainty_penalized",
    "learned_wam",
    "learned_physics_score",
    "learned_energy_regularized",
    "visual_wam",
}

BENCHMARK_SUMMARIES = [
    "benchmark_robocasa_multitask_wam.json",
    "benchmark_robocasa_broad_wam.json",
    "benchmark_robocasa_family12_wam.json",
    "benchmark_robocasa_family24_wam.json",
    "benchmark_robocasa_family28_wam.json",
    "benchmark_robocasa_family32_wam.json",
    "benchmark_robocasa_stratified55_wam.json",
    "benchmark_robocasa_stratified97_wam.json",
    "benchmark_robocasa_residual35_h1_n4_wam.json",
    "benchmark_libero_wam.json",
    "benchmark_metaworld_suite.json",
    "benchmark_robosuite_suite.json",
    "benchmark_maniskill_suite.json",
    "benchmark_gym_robotics_suite.json",
    "benchmark_visual_wam_lite.json",
    "benchmark_gym_robotics_visual_wam.json",
]


@dataclass(frozen=True)
class OptimizerCandidate:
    source: str
    environment: str
    environment_family: str
    model_class: str
    model_capacity: str
    data_scale: int
    rollout_horizon: int | None
    scorer: str
    safety_policy: str
    rollout_budget: int
    objective: str
    mean_delta_vs_random: float | None
    ci_lo_delta_vs_random: float | None
    ci_hi_delta_vs_random: float | None
    evidence_score: float
    status: str

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        out = int(float(value))
    except (TypeError, ValueError):
        return default
    return out


def _ci_status(ci: dict[str, Any]) -> str:
    lo = _safe_float(ci.get("lo"))
    mean = _safe_float(ci.get("mean"))
    if lo is not None and lo > 0.0:
        return "verified_positive"
    if mean is not None and mean > 0.0:
        return "positive_mean_ci_crosses_zero"
    if mean is not None:
        return "nonpositive"
    return "missing_ci"


def _safety_policy(scorer: str) -> str:
    if "safety" in scorer:
        return "safety_penalized"
    if "energy" in scorer:
        return "energy_regularized"
    if "uncertainty" in scorer:
        return "uncertainty_penalized"
    return "none"


def _capacity_for_backend(model_class: str) -> str:
    if model_class == "ensemble_wam":
        return "ridge_ensemble"
    if model_class == "horizon_wam":
        return "ridge_sequence_horizon"
    if model_class == "mlp_dynamics_wam":
        return "mlp_state_action"
    if "visual" in model_class:
        return "visual_feature_model"
    if "task_conditioned" in model_class:
        return "task_conditioned_ridge"
    if "ridge" in model_class:
        return "ridge_linear"
    return model_class or "unknown"


def _family(env: str, source: str) -> str:
    if env.startswith("robocasa/") or "robocasa" in source:
        return "RoboCasa"
    if env.startswith("libero") or "libero" in source:
        return "LIBERO"
    if "metaworld" in source:
        return "Meta-World"
    if "robosuite" in source:
        return "RoboSuite"
    if "maniskill" in source:
        return "ManiSkill"
    if "gym_robotics" in source:
        return "Gymnasium-Robotics"
    if "visual" in source:
        return "VisualToyOrBenchmark"
    return "ToyCPU"


def _toy_train_samples(multi_env: dict[str, Any]) -> dict[tuple[str, str], int]:
    out: dict[tuple[str, str], int] = {}
    for row in multi_env.get("model_metrics") or []:
        if not isinstance(row, dict) or row.get("split") != "train":
            continue
        env = str(row.get("env") or "")
        model = str(row.get("model") or "")
        samples = _safe_int(row.get("n_samples"))
        if env and model and samples > 0:
            out[(env, model)] = samples
    return out


def _toy_candidates(root: Path, results_dir: Path, multi_env: dict[str, Any]) -> list[OptimizerCandidate]:
    rows = _read_csv(results_dir / "tables" / "multi_env_curves.csv")
    if not rows:
        return []

    train_samples = _toy_train_samples(multi_env)
    grouped: dict[tuple[str, str, str, int], dict[int, float]] = defaultdict(dict)
    for row in rows:
        if row.get("mismatch") != "mild":
            continue
        scorer = str(row.get("scorer") or "")
        backend = str(row.get("backend") or "")
        if scorer not in LEARNED_SCORERS and scorer != "random":
            continue
        value = _safe_float(row.get("normalized_real_utility"))
        seed = _safe_int(row.get("seed"), -1)
        n_value = _safe_int(row.get("N"))
        env = str(row.get("env") or "")
        if value is None or seed < 0 or n_value <= 0 or not env or not backend:
            continue
        grouped[(env, backend, scorer, n_value)][seed] = value

    candidates: list[OptimizerCandidate] = []
    for (env, backend, scorer, n_value), by_seed in grouped.items():
        if scorer == "random" or backend in {"analytic_nominal", "oracle_true"}:
            continue
        random_by_seed = grouped.get((env, backend, "random", n_value), {})
        common = sorted(set(by_seed) & set(random_by_seed))
        if len(common) < 2:
            continue
        deltas = [by_seed[seed] - random_by_seed[seed] for seed in common]
        ci = bootstrap_ci(deltas, seed=17, n_boot=2000)
        lo = _safe_float(ci.get("lo"))
        mean = _safe_float(ci.get("mean"))
        hi = _safe_float(ci.get("hi"))
        cost_penalty = 0.015 * math.log2(max(1, n_value))
        score = (lo if lo is not None else -1.0) - cost_penalty
        if scorer == "safety_penalized":
            score += 0.01
        candidates.append(
            OptimizerCandidate(
                source="results/tables/multi_env_curves.csv",
                environment=env,
                environment_family="ToyCPU",
                model_class=backend,
                model_capacity=_capacity_for_backend(backend),
                data_scale=train_samples.get((env, backend), 0),
                rollout_horizon=TOY_HORIZON_BY_ENV.get(env),
                scorer=scorer,
                safety_policy=_safety_policy(scorer),
                rollout_budget=n_value,
                objective="normalized_real_utility_delta_vs_random_minus_log_budget_penalty",
                mean_delta_vs_random=mean,
                ci_lo_delta_vs_random=lo,
                ci_hi_delta_vs_random=hi,
                evidence_score=float(score),
                status=_ci_status(ci),
            )
        )
    return candidates


def _first_ci(summary: dict[str, Any], n_value: int) -> tuple[str, dict[str, Any]]:
    ci_payload = summary.get("confidence_intervals") if isinstance(summary.get("confidence_intervals"), dict) else {}
    preferred = [
        f"best_learned_minus_random_N{n_value}",
        f"promoted_learned_minus_random_N{n_value}",
        f"learned_minus_random_N{n_value}",
        f"visual_minus_random_N{n_value}",
        f"closed_loop_learned_minus_random_N{n_value}",
        f"closed_loop_learned_minus_random_utility_N{n_value}",
        f"learned_energy_regularized_minus_random_N{n_value}",
        f"learned_wam_minus_random_N{n_value}",
    ]
    for key in preferred:
        value = ci_payload.get(key)
        if isinstance(value, dict):
            return key, value
    for key, value in ci_payload.items():
        if isinstance(value, dict) and f"N{n_value}" in str(key) and "random" in str(key):
            return str(key), value
    return "", {}


def _benchmark_candidates(results_dir: Path) -> list[OptimizerCandidate]:
    candidates: list[OptimizerCandidate] = []
    for name in BENCHMARK_SUMMARIES:
        summary = _load_json(results_dir / name)
        if not summary or summary.get("verified") is False:
            continue
        n_values = [_safe_int(value) for value in summary.get("n_values") or []]
        n_values = [value for value in n_values if value > 0]
        n_value = max(n_values) if n_values else _safe_int(summary.get("max_n"))
        if n_value <= 0:
            continue
        ci_key, ci = _first_ci(summary, n_value)
        if not ci:
            continue
        lo = _safe_float(ci.get("lo"))
        mean = _safe_float(ci.get("mean"))
        hi = _safe_float(ci.get("hi"))
        score = (lo if lo is not None else -1.0) - 0.02 * math.log2(max(1, n_value))
        env_ids = summary.get("env_ids") or summary.get("tasks") or summary.get("task_keys") or []
        if isinstance(env_ids, str):
            env_ids = [env_ids]
        env = str(env_ids[0]) if env_ids else str(summary.get("env_id") or summary.get("experiment") or name)
        model_class = str(summary.get("model_type") or summary.get("model") or summary.get("experiment") or name)
        scorer = str(summary.get("promoted_scorer") or ci_key.split("_minus_random")[0] or "learned_wam")
        candidates.append(
            OptimizerCandidate(
                source=f"results/{name}",
                environment=env,
                environment_family=_family(env, name),
                model_class=model_class,
                model_capacity=_capacity_for_backend(model_class),
                data_scale=_safe_int(summary.get("train_samples")) or _safe_int(summary.get("eval_samples")),
                rollout_horizon=_safe_int(summary.get("horizon"), 0) or _safe_int(summary.get("max_horizon"), 0) or None,
                scorer=scorer,
                safety_policy=_safety_policy(scorer),
                rollout_budget=n_value,
                objective=f"{ci_key}_minus_log_budget_penalty",
                mean_delta_vs_random=mean,
                ci_lo_delta_vs_random=lo,
                ci_hi_delta_vs_random=hi,
                evidence_score=float(score),
                status=_ci_status(ci),
            )
        )
    return candidates


def _best_by_environment(candidates: list[OptimizerCandidate]) -> list[OptimizerCandidate]:
    best: dict[str, OptimizerCandidate] = {}
    for candidate in candidates:
        current = best.get(candidate.environment)
        if current is None or candidate.evidence_score > current.evidence_score:
            best[candidate.environment] = candidate
    return sorted(best.values(), key=lambda item: (item.environment_family, item.environment))


def optimize_train_inference(root: Path, results_dir: Path | None = None) -> dict[str, Any]:
    root = root.resolve()
    results_dir = (results_dir or root / "results").resolve()
    multi_env = _load_json(results_dir / "multi_env_suite.json")

    candidates = _toy_candidates(root, results_dir, multi_env)
    candidates.extend(_benchmark_candidates(results_dir))
    selected = _best_by_environment(candidates)

    families = sorted({candidate.environment_family for candidate in selected})
    envs = sorted({candidate.environment for candidate in selected})
    model_classes = sorted({candidate.model_class for candidate in candidates})
    model_capacities = sorted({candidate.model_capacity for candidate in candidates})
    data_scales = sorted({candidate.data_scale for candidate in candidates if candidate.data_scale > 0})
    horizons = sorted({candidate.rollout_horizon for candidate in candidates if candidate.rollout_horizon})
    scorers = sorted({candidate.scorer for candidate in candidates})
    safety_policies = sorted({candidate.safety_policy for candidate in candidates})
    budgets = sorted({candidate.rollout_budget for candidate in candidates if candidate.rollout_budget > 0})

    dimensions = {
        "data_scale": len(data_scales) >= 2,
        "model_class": len(model_classes) >= 3,
        "model_capacity": len(model_capacities) >= 3,
        "rollout_horizon": len(horizons) >= 2,
        "scorer": len(scorers) >= 3,
        "safety_policy": any(policy in {"safety_penalized", "energy_regularized"} for policy in safety_policies),
        "rollout_budget": len(budgets) >= 4,
    }
    checks = [
        {"name": "candidates_present", "ok": len(candidates) >= 10, "detail": f"candidates={len(candidates)}"},
        {"name": "selected_envs_present", "ok": len(selected) >= 5, "detail": f"selected_envs={len(selected)}"},
        {"name": "environment_families_present", "ok": len(families) >= 2, "detail": f"families={families}"},
        {"name": "all_choice_dimensions_covered", "ok": all(dimensions.values()), "detail": f"dimensions={dimensions}"},
    ]
    verified = all(check["ok"] for check in checks)
    global_best = max(candidates, key=lambda item: item.evidence_score) if candidates else None

    return {
        "experiment": "universal_wam_train_inference_optimizer",
        "verified": bool(verified),
        "scope": "evidence_bound_empirical_optimizer_over_existing_artifacts",
        "not_a_universal_proof": True,
        "n_candidates": len(candidates),
        "n_selected_environments": len(selected),
        "n_environment_families": len(families),
        "environment_families": families,
        "environments": envs,
        "model_classes": model_classes,
        "model_capacities": model_capacities,
        "data_scales": data_scales,
        "rollout_horizons": horizons,
        "scorers": scorers,
        "safety_policies": safety_policies,
        "rollout_budgets": budgets,
        "choice_dimensions": dimensions,
        "global_recommendation": global_best.as_dict() if global_best else None,
        "selected_by_environment": [candidate.as_dict() for candidate in selected],
        "candidate_sample": [candidate.as_dict() for candidate in sorted(candidates, key=lambda item: item.evidence_score, reverse=True)[:20]],
        "checks": checks,
        "n_checks": len(checks),
        "n_issues": sum(1 for check in checks if not check["ok"]),
        "limitations": [
            "This optimizer chooses among configurations already represented in committed artifacts; it is not a new universal training law.",
            "The objective is conservative CI-lower-bound improvement over random with a rollout-budget penalty, not a task-agnostic proof of optimality.",
            "Real robot, modern VLA LIBERO, full RoboCasa-wide, and ManiSkill RGB/EE blockers remain separate future-only frontiers.",
        ],
    }


def optimizer_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Train/Inference Optimizer",
        "",
        f"- Verified: `{payload.get('verified')}`",
        f"- Scope: `{payload.get('scope')}`",
        f"- Not a universal proof: `{payload.get('not_a_universal_proof')}`",
        f"- Candidates: `{payload.get('n_candidates')}`",
        f"- Selected environments: `{payload.get('n_selected_environments')}`",
        f"- Environment families: `{payload.get('environment_families')}`",
        "",
        "## Choice Dimensions",
        "",
    ]
    for name, ok in (payload.get("choice_dimensions") or {}).items():
        lines.append(f"- `{name}`: `{ok}`")
    best = payload.get("global_recommendation") or {}
    if best:
        lines.extend(
            [
                "",
                "## Global Conservative Recommendation",
                "",
                f"- Environment: `{best.get('environment')}`",
                f"- Family: `{best.get('environment_family')}`",
                f"- Model class: `{best.get('model_class')}`",
                f"- Data scale: `{best.get('data_scale')}`",
                f"- Horizon: `{best.get('rollout_horizon')}`",
                f"- Scorer: `{best.get('scorer')}`",
                f"- Safety policy: `{best.get('safety_policy')}`",
                f"- Rollout budget: `{best.get('rollout_budget')}`",
                f"- CI lower delta vs random: `{best.get('ci_lo_delta_vs_random')}`",
                f"- Evidence score: `{best.get('evidence_score')}`",
            ]
        )
    lines.extend(["", "## Limitations", ""])
    for item in payload.get("limitations") or []:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)
