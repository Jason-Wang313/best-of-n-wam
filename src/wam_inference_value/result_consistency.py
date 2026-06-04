from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SELF_OUTPUTS = {"result_consistency.json", "artifact_integrity.json", "claims_status.json"}
CORE_BENCHMARKS = {
    "benchmark_maniskill_suite.json": ("env_ids", "benchmark", "benchmark"),
    "benchmark_gym_robotics_suite.json": ("env_ids", "benchmark", "benchmark"),
    "benchmark_metaworld_suite.json": ("task_names", "benchmark", "benchmark"),
    "benchmark_robosuite_suite.json": ("env_names", "benchmark", "benchmark"),
}
ROBOCASA_WAM_RESULTS = [
    "benchmark_robocasa_learned_wam.json",
    "benchmark_robocasa_multitask_wam.json",
    "benchmark_robocasa_broad_wam.json",
    "benchmark_robocasa_family12_wam.json",
    "benchmark_robocasa_family24_wam.json",
    "benchmark_robocasa_extra4_wam.json",
    "benchmark_robocasa_family28_wam.json",
    "benchmark_robocasa_family32_wam.json",
    "benchmark_robocasa_stratified55_wam.json",
    "benchmark_robocasa_stratified97_wam.json",
    "benchmark_robocasa_residual35_h1_n4_wam.json",
]
LIBERO_POLICY_RESULTS = [
    "benchmark_libero_scripted_policy.json",
    "benchmark_libero_learned_action_head.json",
    "benchmark_libero_autonomous_bc_policy.json",
    "benchmark_libero_visual_language_bc_policy.json",
]


@dataclass(frozen=True)
class ConsistencyCheck:
    name: str
    ok: bool
    detail: str


def resolve_path(value: Any, root: Path) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def load_json(results_dir: Path, name: str) -> dict[str, Any]:
    path = results_dir / name
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(path_value: Any, root: Path) -> list[dict[str, str]]:
    path = resolve_path(path_value, root)
    if path is None or not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def add(checks: list[ConsistencyCheck], name: str, ok: bool, detail: str) -> None:
    checks.append(ConsistencyCheck(name=name, ok=bool(ok), detail=detail))


def str_set(values: Any) -> set[str]:
    if isinstance(values, str):
        return {values}
    if not isinstance(values, list):
        return set()
    return {str(value) for value in values}


def unique_values(rows: list[dict[str, str]], field: str) -> set[str]:
    return {row.get(field, "") for row in rows if row.get(field, "") != ""}


def unique_ints(rows: list[dict[str, str]], field: str) -> set[int]:
    values: set[int] = set()
    for row in rows:
        raw = row.get(field, "")
        if raw == "":
            continue
        values.add(int(float(raw)))
    return values


def row_count(path_value: Any, root: Path) -> int:
    return len(read_csv_rows(path_value, root))


def pool_count(rows: list[dict[str, str]], id_field: str) -> int:
    return len({(row.get(id_field), row.get("seed"), row.get("state_id")) for row in rows})


def ci_objects(value: Any, trail: tuple[str, ...] = ()) -> list[tuple[str, dict[str, Any]]]:
    found: list[tuple[str, dict[str, Any]]] = []
    if isinstance(value, dict):
        keys = set(value)
        if {"n", "mean", "lo", "hi"}.issubset(keys):
            found.append((".".join(trail), value))
        for key, nested in value.items():
            found.extend(ci_objects(nested, trail + (str(key),)))
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            found.extend(ci_objects(nested, trail + (str(index),)))
    return found


def audit_ci_sanity(root: Path, results_dir: Path, checks: list[ConsistencyCheck]) -> None:
    total = 0
    bad: list[str] = []
    for path in sorted(results_dir.glob("*.json")):
        if path.name in SELF_OUTPUTS:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover - covered by artifact audit.
            bad.append(f"{path.name}: invalid JSON: {exc}")
            continue
        for trail, ci in ci_objects(payload):
            total += 1
            try:
                n = int(ci["n"])
                mean = float(ci["mean"])
                lo = float(ci["lo"])
                hi = float(ci["hi"])
                std = float(ci.get("std", 0.0))
                stderr = float(ci.get("stderr", 0.0))
                ci95 = float(ci.get("ci95", 0.0))
            except (TypeError, ValueError) as exc:
                bad.append(f"{path.name}:{trail}: nonnumeric CI field: {exc}")
                continue
            if n < 1:
                bad.append(f"{path.name}:{trail}: n < 1")
            if lo > mean or mean > hi:
                bad.append(f"{path.name}:{trail}: mean not inside [lo, hi]")
            if std < 0.0 or stderr < 0.0 or ci95 < 0.0:
                bad.append(f"{path.name}:{trail}: negative uncertainty field")
    add(checks, "confidence_interval_sanity", total >= 250 and not bad, f"ci_objects={total}, issues={len(bad)}")
    for issue in bad[:25]:
        add(checks, f"confidence_interval_issue:{issue}", False, issue)


def audit_multi_env(root: Path, results_dir: Path, checks: list[ConsistencyCheck]) -> None:
    payload = load_json(results_dir, "multi_env_suite.json")
    artifacts = payload.get("artifacts") or {}
    curves = read_csv_rows(artifacts.get("curves"), root)
    aggregate = read_csv_rows(artifacts.get("aggregate"), root)
    envs = str_set(payload.get("envs"))
    backbones = str_set(payload.get("backbones"))
    expected_backends = backbones | {"analytic_nominal", "oracle_true"}
    seeds = {int(seed) for seed in payload.get("seeds", [])}
    n_values = {int(value) for value in payload.get("N_values", [])}
    model_metrics = payload.get("model_metrics") or []
    inference_claims = payload.get("inference_claims") or []
    mismatch_claims = payload.get("mismatch_gap_claims") or []

    add(checks, "multi_env_curves_present", len(curves) >= 1000, f"rows={len(curves)}")
    add(checks, "multi_env_env_coverage", unique_values(curves, "env") == envs, f"table={sorted(unique_values(curves, 'env'))}, json={sorted(envs)}")
    add(
        checks,
        "multi_env_backend_coverage",
        unique_values(curves, "backend") == expected_backends,
        f"table={sorted(unique_values(curves, 'backend'))}, expected={sorted(expected_backends)}",
    )
    add(checks, "multi_env_seed_coverage", unique_ints(curves, "seed") == seeds, f"table={sorted(unique_ints(curves, 'seed'))}, json={sorted(seeds)}")
    add(checks, "multi_env_n_coverage", unique_ints(curves, "N") == n_values, f"table={sorted(unique_ints(curves, 'N'))}, json={sorted(n_values)}")
    add(checks, "multi_env_model_metric_count", len(model_metrics) == len(envs) * len(backbones) * 3, f"rows={len(model_metrics)}")
    missing_metrics = []
    for env in envs:
        for model in backbones:
            splits = {row.get("split") for row in model_metrics if row.get("env") == env and row.get("model") == model}
            if splits != {"train", "validation", "ood_severe"}:
                missing_metrics.append(f"{env}/{model}:{sorted(splits)}")
    add(checks, "multi_env_model_metric_split_coverage", not missing_metrics, f"missing={missing_metrics[:5]}")
    expected_claim_count = len(envs) * len(expected_backends)
    add(checks, "multi_env_inference_claim_count", len(inference_claims) == expected_claim_count, f"claims={len(inference_claims)}, expected={expected_claim_count}")
    ci_n_ok = all((claim.get("n64_minus_n1") or {}).get("n") == len(seeds) for claim in inference_claims)
    add(checks, "multi_env_inference_claim_ci_n", ci_n_ok, f"claims={len(inference_claims)}, n_seeds={len(seeds)}")
    expected_gap_count = len(envs) * (len(backbones) + 1)
    gap_n_ok = all((claim.get("severe_minus_mild_gap") or {}).get("n") == len(seeds) for claim in mismatch_claims)
    add(checks, "multi_env_mismatch_claim_count", len(mismatch_claims) == expected_gap_count, f"claims={len(mismatch_claims)}, expected={expected_gap_count}")
    add(checks, "multi_env_mismatch_claim_ci_n", gap_n_ok, f"claims={len(mismatch_claims)}, n_seeds={len(seeds)}")
    add(checks, "multi_env_aggregate_present", len(aggregate) >= 100, f"rows={len(aggregate)}")


def audit_core_benchmark(root: Path, results_dir: Path, checks: list[ConsistencyCheck], name: str, ids_key: str, table_id_field: str, label: str) -> None:
    payload = load_json(results_dir, name)
    artifacts = payload.get("artifacts") or {}
    curves = read_csv_rows(artifacts.get("curves"), root)
    exact = read_csv_rows(artifacts.get("exact_law"), root)
    closed = read_csv_rows(artifacts.get("closed_loop"), root)
    ids = str_set(payload.get(ids_key))
    if ids_key == "task_names" and not ids:
        ids = str_set(payload.get("task_names"))
    if ids_key == "env_names" and not ids:
        ids = str_set(payload.get("env_names"))
    table_ids = unique_values(curves, table_id_field)
    if ids_key == "task_names":
        ids = str_set(payload.get("task_names"))
        table_ids = unique_values(curves, "benchmark")
    if ids_key == "env_names":
        ids = str_set(payload.get("env_names"))
        table_ids = unique_values(curves, "benchmark")
    json_pools = int(payload.get("n_rollout_pools") or 0)
    json_seeds = {int(seed) for seed in payload.get("seeds", [])} if payload.get("seeds") else unique_ints(curves, "seed")
    expected_n = {1, 2, 4, 8, 16, 32}

    add(checks, f"{label}_curves_present", len(curves) >= 500, f"rows={len(curves)}")
    add(checks, f"{label}_id_coverage", table_ids == ids, f"table={sorted(table_ids)}, json={sorted(ids)}")
    add(checks, f"{label}_pool_count", pool_count(curves, table_id_field) == json_pools, f"table={pool_count(curves, table_id_field)}, json={json_pools}")
    add(checks, f"{label}_seed_count", len(unique_ints(curves, "seed")) >= 5 and unique_ints(curves, "seed") == json_seeds, f"seeds={sorted(unique_ints(curves, 'seed'))}")
    add(checks, f"{label}_n_values", unique_ints(curves, "N") == expected_n, f"N={sorted(unique_ints(curves, 'N'))}")
    add(checks, f"{label}_exact_rows", len(exact) >= 100, f"rows={len(exact)}")
    if artifacts.get("closed_loop"):
        add(checks, f"{label}_closed_loop_rows", len(closed) >= 50, f"rows={len(closed)}")


def audit_visual_benchmark(root: Path, results_dir: Path, checks: list[ConsistencyCheck]) -> None:
    for name, label, min_rows in [
        ("benchmark_visual_wam_lite.json", "benchmark_rgb_visual_wam", 500),
        ("benchmark_gym_robotics_visual_wam.json", "gym_robotics_rgb_visual_wam", 1000),
    ]:
        payload = load_json(results_dir, name)
        artifacts = payload.get("artifacts") or {}
        curves = read_csv_rows(artifacts.get("curves") or artifacts.get("table"), root)
        exact = read_csv_rows(artifacts.get("exact_law"), root)
        add(checks, f"{label}_curves_present", len(curves) >= min_rows, f"rows={len(curves)}")
        add(checks, f"{label}_exact_rows", len(exact) >= 100, f"rows={len(exact)}")
        if payload.get("env_ids"):
            add(checks, f"{label}_env_coverage", unique_values(curves, "benchmark") == str_set(payload.get("env_ids")), f"envs={sorted(unique_values(curves, 'benchmark'))}")
        if payload.get("model_metrics"):
            model_paths = [row.get("model_path") for row in payload.get("model_metrics") or [] if row.get("model_path")]
            frame_paths = [row.get("frame_path") for row in payload.get("model_metrics") or [] if row.get("frame_path")]
            add(checks, f"{label}_model_artifacts", len(model_paths) >= 3 and all((resolve_path(path, root) or Path()).exists() for path in model_paths), f"models={len(model_paths)}")
            add(checks, f"{label}_frame_artifacts", len(frame_paths) >= 3 and all((resolve_path(path, root) or Path()).exists() for path in frame_paths), f"frames={len(frame_paths)}")


def audit_rollout_pool_wam(root: Path, results_dir: Path, checks: list[ConsistencyCheck], name: str, label: str) -> None:
    payload = load_json(results_dir, name)
    curves = read_csv_rows(payload.get("curves_path"), root)
    exact = read_csv_rows(payload.get("exact_path"), root)
    data = read_csv_rows(payload.get("data_path"), root)
    eval_rows = read_csv_rows(payload.get("eval_path"), root)
    seed_rows = read_csv_rows(payload.get("seed_metrics_path"), root)
    id_field = "env_id" if curves and "env_id" in curves[0] else "task_key"
    declared_ids = str_set(payload.get("env_ids") or payload.get("tasks") or payload.get("env_id"))
    n_values = {int(value) for value in payload.get("n_values", [])}
    expected_pools = int(payload.get("eval_rollout_pools") or payload.get("eval_states") or 0)
    expected_eval = int(payload.get("eval_samples") or 0)
    expected_data = int(payload.get("train_samples") or 0) + int(payload.get("validation_samples") or 0)
    ci = payload.get("confidence_intervals") or {}
    promoted = payload.get("promoted_scorer")
    promoted_ci = None
    if promoted:
        promoted_ci = ci.get(f"{promoted}_minus_random_N8") or ci.get("best_learned_minus_random_N8")

    add(checks, f"{label}_eval_rows", len(eval_rows) == expected_eval, f"rows={len(eval_rows)}, json={expected_eval}")
    add(checks, f"{label}_data_rows", len(data) == expected_data, f"rows={len(data)}, json={expected_data}")
    add(checks, f"{label}_pool_count", pool_count(curves, id_field) == expected_pools, f"table={pool_count(curves, id_field)}, json={expected_pools}")
    add(checks, f"{label}_id_coverage", unique_values(curves, id_field) == declared_ids, f"table={len(unique_values(curves, id_field))}, json={len(declared_ids)}")
    add(checks, f"{label}_n_values", unique_ints(curves, "N") == n_values, f"N={sorted(unique_ints(curves, 'N'))}, json={sorted(n_values)}")
    add(checks, f"{label}_exact_rows", len(exact) >= max(1, expected_pools * len(n_values)), f"rows={len(exact)}, pools={expected_pools}")
    add(checks, f"{label}_seed_metrics_rows", len(seed_rows) >= max(1, expected_pools), f"rows={len(seed_rows)}, pools={expected_pools}")
    if promoted_ci is not None:
        add(checks, f"{label}_promoted_ci_positive", float(promoted_ci.get("lo", -1.0)) > 0.0, f"promoted={promoted}, ci={promoted_ci}")


def audit_libero_policies(root: Path, results_dir: Path, checks: list[ConsistencyCheck]) -> None:
    for name in LIBERO_POLICY_RESULTS:
        payload = load_json(results_dir, name)
        artifact_paths = payload.get("artifact_paths") or {}
        rows = read_csv_rows(artifact_paths.get("episodes_csv"), root)
        label = name.removesuffix(".json")
        tasks = str_set(payload.get("tasks"))
        task_ids = unique_values(rows, "task_id")
        add(checks, f"{label}_episode_rows", len(rows) >= int(payload.get("eval_episodes") or payload.get("episodes") or 0), f"rows={len(rows)}")
        add(checks, f"{label}_task_coverage", task_ids == tasks, f"table={len(task_ids)}, json={len(tasks)}")
        if "eval_episodes" in payload:
            eval_rows = [row for row in rows if row.get("split", "").startswith("eval")]
            successes = sum(1 for row in eval_rows if str(row.get("success", "")).lower() == "true")
            add(checks, f"{label}_eval_episode_count", len(eval_rows) == int(payload.get("eval_episodes") or 0), f"rows={len(eval_rows)}, json={payload.get('eval_episodes')}")
            add(checks, f"{label}_eval_success_count", successes == int(payload.get("eval_successes") or 0), f"successes={successes}, json={payload.get('eval_successes')}")
        else:
            successes = sum(1 for row in rows if str(row.get("success", "")).lower() == "true")
            add(checks, f"{label}_success_count", successes == int(payload.get("successes") or successes), f"successes={successes}")


def audit_result_consistency(root: Path, results_dir: Path | None = None) -> dict[str, Any]:
    root = root.resolve()
    results_dir = (results_dir or root / "results").resolve()
    checks: list[ConsistencyCheck] = []

    audit_ci_sanity(root, results_dir, checks)
    audit_multi_env(root, results_dir, checks)
    for name, (ids_key, table_id_field, label) in CORE_BENCHMARKS.items():
        audit_core_benchmark(root, results_dir, checks, name, ids_key, table_id_field, label)
    audit_visual_benchmark(root, results_dir, checks)
    for name in ROBOCASA_WAM_RESULTS:
        audit_rollout_pool_wam(root, results_dir, checks, name, name.removesuffix(".json"))
    audit_rollout_pool_wam(root, results_dir, checks, "benchmark_libero_wam.json", "benchmark_libero_wam")
    audit_libero_policies(root, results_dir, checks)

    issues = [check for check in checks if not check.ok]
    return {
        "verified": len(issues) == 0,
        "n_checks": len(checks),
        "n_issues": len(issues),
        "checks": [check.__dict__ for check in checks],
        "issues": [check.__dict__ for check in issues],
    }


def result_consistency_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Result Consistency Report",
        "",
        f"- Verified: {payload.get('verified')}",
        f"- Checks: {payload.get('n_checks')}",
        f"- Issues: {payload.get('n_issues')}",
        "",
    ]
    issues = payload.get("issues") or []
    if issues:
        lines.append("## Issues")
        lines.append("")
        for issue in issues:
            lines.append(f"- `{issue.get('name')}`: {issue.get('detail')}")
    else:
        lines.append("Summary JSONs agree with their canonical tables for CI sanity, row counts, seed coverage, task/environment coverage, rollout-pool counts, promoted-scorer CIs, and LIBERO success counts.")
    lines.append("")
    return "\n".join(lines)
