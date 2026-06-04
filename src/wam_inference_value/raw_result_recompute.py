from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from wam_inference_value.evaluation import ci95


SELF_OUTPUTS = {
    "artifact_integrity.json",
    "claim_evidence_quality.json",
    "claim_generation_consistency.json",
    "claim_ledger_integrity.json",
    "claims_status.json",
    "command_result_consistency.json",
    "experiment_registry.json",
    "model_artifact_integrity.json",
    "narrative_consistency.json",
    "raw_result_recompute.json",
    "report_generation_consistency.json",
    "result_consistency.json",
    "script_contracts.json",
    "test_inventory.json",
}


@dataclass(frozen=True)
class RawRecomputeCheck:
    name: str
    ok: bool
    detail: str


def add(checks: list[RawRecomputeCheck], name: str, ok: bool, detail: str) -> None:
    checks.append(RawRecomputeCheck(name=name, ok=bool(ok), detail=detail))


def resolve_path(root: Path, raw_path: Any) -> Path | None:
    if not isinstance(raw_path, str) or not raw_path:
        return None
    path = Path(raw_path).expanduser()
    return path if path.is_absolute() else root / path


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def read_csv_rows(root: Path, raw_path: Any) -> list[dict[str, str]]:
    path = resolve_path(root, raw_path)
    if path is None or not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def key_value(row: dict[str, Any], key: str) -> str:
    value = row.get(key)
    if key in {"N", "seed", "state_id", "task_index"}:
        parsed = finite_float(value)
        if parsed is not None:
            return str(int(parsed))
    return str(value)


def grouped_means(rows: list[dict[str, str]], keys: list[str], metrics: list[str]) -> dict[tuple[str, ...], dict[str, float]]:
    sums: dict[tuple[str, ...], dict[str, float]] = {}
    counts: dict[tuple[str, ...], dict[str, int]] = {}
    for row in rows:
        group = tuple(key_value(row, key) for key in keys)
        sums.setdefault(group, {})
        counts.setdefault(group, {})
        for metric in metrics:
            value = finite_float(row.get(metric))
            if value is None:
                continue
            sums[group][metric] = sums[group].get(metric, 0.0) + value
            counts[group][metric] = counts[group].get(metric, 0) + 1
    means: dict[tuple[str, ...], dict[str, float]] = {}
    for group, group_sums in sums.items():
        means[group] = {}
        for metric, value_sum in group_sums.items():
            count = counts[group].get(metric, 0)
            if count > 0:
                means[group][metric] = value_sum / count
    return means


def compare_grouped_summary(
    checks: list[RawRecomputeCheck],
    *,
    label: str,
    raw_rows: list[dict[str, str]],
    summary_rows: list[dict[str, Any]],
    keys: list[str],
    metrics: list[str],
    min_summary_rows: int,
    tolerance: float = 1e-10,
) -> dict[str, Any]:
    recomputed = grouped_means(raw_rows, keys, metrics)
    summary = {tuple(key_value(row, key) for key in keys): row for row in summary_rows}
    missing = sorted(set(summary) - set(recomputed))
    extra = sorted(set(recomputed) - set(summary))
    max_abs_diff = 0.0
    compared = 0
    bad: list[str] = []
    for group, summary_row in summary.items():
        raw_row = recomputed.get(group)
        if raw_row is None:
            continue
        for metric in metrics:
            expected = finite_float(summary_row.get(metric))
            actual = raw_row.get(metric)
            if expected is None or actual is None:
                continue
            diff = abs(actual - expected)
            max_abs_diff = max(max_abs_diff, diff)
            compared += 1
            if diff > tolerance:
                bad.append(f"{group}:{metric}: raw={actual}, summary={expected}, diff={diff}")
    add(checks, f"{label}_rows_present", len(summary_rows) >= min_summary_rows, f"rows={len(summary_rows)}")
    add(checks, f"{label}_groups_match", not missing and not extra, f"missing={len(missing)}, extra={len(extra)}")
    add(checks, f"{label}_means_match_raw_rows", not bad and compared > 0, f"metrics={compared}, max_abs_diff={max_abs_diff}")
    return {
        "label": label,
        "summary_rows": len(summary_rows),
        "groups": len(summary),
        "metrics_compared": compared,
        "max_abs_diff": max_abs_diff,
        "mismatches": bad[:25],
    }


def audit_aggregate_recomputes(root: Path, results_dir: Path, checks: list[RawRecomputeCheck]) -> dict[str, Any]:
    learned = load_json(results_dir / "learned_wam_vs_analytic_wam.json")
    learned_artifacts = learned.get("artifacts") or {}
    learned_table = read_csv_rows(root, learned_artifacts.get("table"))
    learned_summary = learned.get("aggregate") or []
    learned_record = compare_grouped_summary(
        checks,
        label="learned_wam_vs_analytic_aggregate",
        raw_rows=learned_table,
        summary_rows=learned_summary,
        keys=["backend", "N"],
        metrics=[
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
        ],
        min_summary_rows=21,
    )

    multi_env = load_json(results_dir / "multi_env_suite.json")
    multi_artifacts = multi_env.get("artifacts") or {}
    multi_curves = read_csv_rows(root, multi_artifacts.get("curves"))
    multi_aggregate = read_csv_rows(root, multi_artifacts.get("aggregate"))
    multi_record = compare_grouped_summary(
        checks,
        label="multi_env_aggregate",
        raw_rows=multi_curves,
        summary_rows=multi_aggregate,
        keys=["env", "backend", "scorer", "mismatch", "N"],
        metrics=[
            "success",
            "real_utility",
            "imagined_utility",
            "gap_imagined_minus_real",
            "normalized_real_utility",
        ],
        min_summary_rows=100,
    )
    return {
        "aggregate_recomputes": [learned_record, multi_record],
        "aggregate_metrics_compared": learned_record["metrics_compared"] + multi_record["metrics_compared"],
    }


def exact_path_from_payload(payload: dict[str, Any]) -> Any:
    artifacts = payload.get("artifacts") or {}
    return artifacts.get("exact_law") or payload.get("exact_path")


def audit_exact_law_mae(root: Path, results_dir: Path, checks: list[RawRecomputeCheck]) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    bad: list[str] = []
    for path in sorted(results_dir.glob("*.json")):
        if path.name in SELF_OUTPUTS:
            continue
        payload = load_json(path)
        expected = finite_float(payload.get("exact_law_utility_mae"))
        exact_path = exact_path_from_payload(payload)
        if expected is None or exact_path is None:
            continue
        rows = read_csv_rows(root, exact_path)
        values = [finite_float(row.get("utility_abs_error")) for row in rows]
        values = [value for value in values if value is not None]
        actual = float(np.mean(values)) if values else None
        diff = abs(actual - expected) if actual is not None else None
        ok = actual is not None and diff is not None and diff <= 1e-10
        record = {
            "json": path.name,
            "rows": len(values),
            "recomputed_exact_law_utility_mae": actual,
            "summary_exact_law_utility_mae": expected,
            "abs_diff": diff,
            "ok": ok,
        }
        records.append(record)
        if not ok:
            bad.append(f"{path.name}: raw={actual}, summary={expected}, diff={diff}")
    add(checks, "exact_law_mae_file_count", len(records) >= 20, f"files={len(records)}")
    add(checks, "exact_law_mae_matches_raw_tables", not bad, f"files={len(records)}, mismatches={len(bad)}")
    return {
        "exact_law_mae_files": len(records),
        "exact_law_mae_mismatches": bad[:25],
        "exact_law_mae_records": records,
    }


def seed_metrics_path_from_payload(payload: dict[str, Any]) -> Any:
    artifacts = payload.get("artifacts") or {}
    return payload.get("seed_metrics_path") or artifacts.get("seed_metrics")


def compare_ci(actual: dict[str, Any], expected: dict[str, Any], tolerance: float = 1e-10) -> bool:
    for key in ["n", "mean", "std", "stderr", "ci95", "lo", "hi"]:
        lhs = actual.get(key)
        rhs = expected.get(key)
        if lhs is None or rhs is None:
            if lhs != rhs:
                return False
            continue
        if abs(float(lhs) - float(rhs)) > tolerance:
            return False
    return True


def audit_seed_metric_cis(root: Path, results_dir: Path, checks: list[RawRecomputeCheck]) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    bad: list[str] = []
    alias_records: list[dict[str, Any]] = []
    alias_bad: list[str] = []
    for path in sorted(results_dir.glob("*.json")):
        if path.name in SELF_OUTPUTS:
            continue
        payload = load_json(path)
        ci = payload.get("confidence_intervals") or {}
        seed_path = seed_metrics_path_from_payload(payload)
        if not ci or seed_path is None:
            continue
        rows = read_csv_rows(root, seed_path)
        if not rows:
            continue
        columns = set(rows[0])
        for key, summary_ci in ci.items():
            if key not in columns or not isinstance(summary_ci, dict) or "mean" not in summary_ci:
                continue
            values = [finite_float(row.get(key)) for row in rows]
            values = [value for value in values if value is not None]
            recomputed = ci95(values)
            ok = compare_ci(recomputed, summary_ci)
            records.append({"json": path.name, "column": key, "n": len(values), "ok": ok})
            if not ok:
                bad.append(f"{path.name}:{key}: raw={recomputed}, summary={summary_ci}")
        n_values = [int(value) for value in payload.get("n_values") or payload.get("N_values") or []]
        if payload.get("promoted_scorer") and n_values:
            n_max = max(n_values)
            promoted_key = f"promoted_learned_minus_random_N{n_max}"
            legacy_key = f"best_learned_minus_random_N{n_max}"
            promoted_ci = ci.get(promoted_key)
            legacy_ci = ci.get(legacy_key)
            if promoted_ci is not None and legacy_ci is not None:
                ok = compare_ci(promoted_ci, legacy_ci)
                alias_records.append({"json": path.name, "promoted_key": promoted_key, "legacy_key": legacy_key, "ok": ok})
                if not ok:
                    alias_bad.append(f"{path.name}:{legacy_key} != {promoted_key}")
    add(checks, "seed_metric_ci_column_count", len(records) >= 120, f"columns={len(records)}")
    add(checks, "seed_metric_cis_match_raw_columns", not bad, f"columns={len(records)}, mismatches={len(bad)}")
    add(checks, "promoted_learned_alias_count", len(alias_records) >= 10, f"aliases={len(alias_records)}")
    add(checks, "promoted_learned_aliases_match", not alias_bad, f"aliases={len(alias_records)}, mismatches={len(alias_bad)}")
    return {
        "seed_metric_ci_columns": len(records),
        "seed_metric_ci_mismatches": bad[:25],
        "seed_metric_ci_records": records,
        "promoted_alias_records": alias_records,
        "promoted_alias_mismatches": alias_bad,
    }


def audit_raw_result_recompute(root: Path, results_dir: Path | None = None) -> dict[str, Any]:
    root = root.resolve()
    results_dir = (results_dir or root / "results").resolve()
    checks: list[RawRecomputeCheck] = []
    aggregate = audit_aggregate_recomputes(root, results_dir, checks)
    exact = audit_exact_law_mae(root, results_dir, checks)
    seed_cis = audit_seed_metric_cis(root, results_dir, checks)
    issues = [check for check in checks if not check.ok]
    return {
        "experiment": "raw_result_recompute",
        "verified": len(issues) == 0,
        "results_dir": str(results_dir),
        "n_checks": len(checks),
        "n_issues": len(issues),
        "checks": [check.__dict__ for check in checks],
        "issues": [check.__dict__ for check in issues],
        **aggregate,
        **exact,
        **seed_cis,
    }


def raw_result_recompute_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Raw Result Recompute Report",
        "",
        f"- Verified: {payload.get('verified')}",
        f"- Checks: {payload.get('n_checks')}",
        f"- Issues: {payload.get('n_issues')}",
        f"- Aggregate metrics recomputed: {payload.get('aggregate_metrics_compared')}",
        f"- Exact-law MAE files recomputed: {payload.get('exact_law_mae_files')}",
        f"- Seed-metric CI columns recomputed: {payload.get('seed_metric_ci_columns')}",
        f"- Promoted learned aliases checked: {len(payload.get('promoted_alias_records') or [])}",
        "",
    ]
    issues = payload.get("issues") or []
    if issues:
        lines.append("## Issues")
        lines.append("")
        for issue in issues:
            lines.append(f"- `{issue.get('name')}`: {issue.get('detail')}")
    else:
        lines.append("Summary aggregates, exact-law MAEs, and seed-metric confidence intervals recompute from raw CSV artifacts.")
    lines.append("")
    return "\n".join(lines)
