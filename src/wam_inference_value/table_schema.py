from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


OPTIONAL_BLANK_COLUMNS = {
    "kappa",
    "failure_reason",
    "error",
    "error_type",
    "mean_utility_variance",
    "initial_distance",
    "mean_final_distance",
    "mean_progress",
    "mean_utility",
    "utility_std",
    "utility_min",
    "utility_max",
    "utility_corr",
    "render_mode",
    "render_backend",
    "sim_backend",
    "shader_dir",
    "sensor_width",
    "sensor_height",
    "human_width",
    "human_height",
    "env_overrides",
    "env_ids",
}
TEXT_COLUMNS = {
    "alignment_status",
    "artifact",
    "backend",
    "benchmark",
    "best_learned_scorer",
    "case",
    "category",
    "control_mode",
    "decision_action",
    "decision_reason",
    "dynamics_backend",
    "env",
    "env_id",
    "env_ids",
    "env_overrides",
    "error",
    "error_type",
    "evidence_level",
    "failure_reason",
    "grasp_profile",
    "mismatch",
    "model",
    "name",
    "object_name",
    "obs_mode",
    "phase",
    "policy",
    "pool_oracle_best_learned_scorer",
    "profile_class",
    "reason",
    "render_mode",
    "render_backend",
    "scorer",
    "shader_dir",
    "sim_backend",
    "split",
    "summary_path",
    "tag",
    "target_name",
    "task_id",
    "task_key",
    "task_name",
}
BOOLEAN_COLUMNS = {
    "available",
    "covered_by_any_committed_artifact",
    "covered_by_micro_rollout_probe",
    "covered_by_verified_rollout_pool_artifact",
    "nondegenerate",
    "ok",
    "present",
    "reset_ok",
    "rollout_ok",
    "timed_out",
    "verified",
}
INTEGER_LIKE_COLUMNS = {
    "K",
    "N",
    "candidate_task_count",
    "chunk_index",
    "count",
    "episode",
    "eval_state_id",
    "feature_dim",
    "feature_rows",
    "heldout_n",
    "horizon",
    "max_horizon",
    "min_horizon",
    "n_env_ids",
    "n_samples",
    "nondegenerate_task_count",
    "num_states",
    "pilot_k",
    "pool_size",
    "recommended_n",
    "returncode",
    "rollout_id",
    "rollout_steps_proxy",
    "rollouts",
    "runnable_task_count",
    "seed",
    "seed_id",
    "split_seed",
    "state_id",
    "steps",
    "stop_n",
    "success_count",
    "t",
    "target_dim",
    "task_index",
}
NUMERIC_TOKENS = (
    "abs_error",
    "auc",
    "corr",
    "delta",
    "distance",
    "energy",
    "gain",
    "gap",
    "height",
    "kappa",
    "mae",
    "mean",
    "minus",
    "p",
    "predicted",
    "progress",
    "rate",
    "reward",
    "score",
    "seconds",
    "std",
    "success",
    "utility",
    "value",
    "variance",
    "width",
)
VALUE_METRIC_COLUMNS = {
    "delta_value",
    "final_distance",
    "gap_imagined_minus_real",
    "imagined_utility",
    "mean_utility",
    "normalized_real_utility",
    "progress",
    "real_utility",
    "success",
    "utility",
}


@dataclass(frozen=True)
class TableSchemaCheck:
    name: str
    ok: bool
    detail: str


def add(checks: list[TableSchemaCheck], name: str, ok: bool, detail: str) -> None:
    checks.append(TableSchemaCheck(name=name, ok=bool(ok), detail=detail))


def table_paths(results_dir: Path) -> list[Path]:
    tables_dir = results_dir / "tables"
    if not tables_dir.exists():
        return []
    return sorted(path for path in tables_dir.glob("*.csv") if path.is_file())


def parse_bool(value: str) -> bool | None:
    lowered = value.strip().lower()
    if lowered in {"true", "1"}:
        return True
    if lowered in {"false", "0"}:
        return False
    return None


def parse_float(value: str) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def is_numeric_column(column: str) -> bool:
    if column in TEXT_COLUMNS:
        return False
    if column in BOOLEAN_COLUMNS or column in INTEGER_LIKE_COLUMNS:
        return True
    lower = column.lower()
    return any(token in lower for token in NUMERIC_TOKENS)


def expected_family_requirements(name: str) -> list[tuple[str, set[str], str]]:
    requirements: list[tuple[str, set[str], str]] = []
    if "curves" in name:
        requirements.append(("curves_have_N", {"N"}, "curves tables must include N"))
        requirements.append(("curves_have_value_metric", VALUE_METRIC_COLUMNS, "curves tables must include at least one value metric"))
    if "exact_law" in name or "exact_rollout_law" in name:
        requirements.append(("exact_law_has_N", {"N"}, "exact-law tables must include N"))
        requirements.append(("exact_law_has_error_metric", {"utility_abs_error", "success_abs_error", "abs_error"}, "exact-law tables must include an error metric"))
    if "seed_metrics" in name:
        requirements.append(("seed_metrics_have_seed", {"seed"}, "seed metric tables must include seed"))
    if "train_validation" in name:
        requirements.append(("train_validation_has_split", {"split"}, "train/validation tables must include split"))
        requirements.append(("train_validation_has_seed", {"seed"}, "train/validation tables must include seed"))
    if "eval_rollouts" in name:
        requirements.append(("eval_rollouts_have_rollout_id", {"rollout_id"}, "eval rollout tables must include rollout_id"))
        requirements.append(("eval_rollouts_have_value", {"utility", "success", "progress"}, "eval rollout tables must include utility/success/progress"))
    if "closed_loop" in name and "aggregate" not in name:
        requirements.append(("closed_loop_has_seed", {"seed"}, "closed-loop tables must include seed"))
    if "closed_loop" in name:
        requirements.append(("closed_loop_has_N", {"N"}, "closed-loop tables must include N"))
        requirements.append(("closed_loop_has_scorer", {"scorer"}, "closed-loop tables must include scorer"))
    return requirements


def audit_table(path: Path) -> dict[str, Any]:
    malformed = False
    duplicate_rows = 0
    blank_disallowed: list[str] = []
    nonfinite_numeric: list[str] = []
    noninteger_values: list[str] = []
    negative_id_values: list[str] = []
    nonpositive_compute_values: list[str] = []
    success_range_errors: list[str] = []
    boolean_errors: list[str] = []
    numeric_columns: set[str] = set()
    blank_columns: set[str] = set()
    seen_rows: set[tuple[str, ...]] = set()

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    for row_index, row in enumerate(rows, start=2):
        if None in row:
            malformed = True
        raw_tuple = tuple(str(row.get(column, "")) for column in fieldnames)
        if raw_tuple in seen_rows:
            duplicate_rows += 1
        seen_rows.add(raw_tuple)
        if not any(str(row.get(column, "")).strip() for column in fieldnames):
            malformed = True
        for column in fieldnames:
            value = row.get(column)
            value_text = "" if value is None else str(value).strip()
            if value_text == "":
                blank_columns.add(column)
                if column not in OPTIONAL_BLANK_COLUMNS:
                    blank_disallowed.append(f"{row_index}:{column}")
                continue
            if column in BOOLEAN_COLUMNS:
                if parse_bool(value_text) is None:
                    boolean_errors.append(f"{row_index}:{column}={value_text}")
                continue
            if not is_numeric_column(column):
                continue
            bool_value = parse_bool(value_text)
            parsed = float(bool_value) if bool_value is not None else parse_float(value_text)
            if parsed is None:
                nonfinite_numeric.append(f"{row_index}:{column}={value_text}")
                continue
            numeric_columns.add(column)
            if column in INTEGER_LIKE_COLUMNS and not parsed.is_integer():
                noninteger_values.append(f"{row_index}:{column}={value_text}")
            lower = column.lower()
            if column in {"seed", "seed_id", "state_id", "eval_state_id", "rollout_id", "task_index", "episode", "t", "count"} and parsed < 0:
                negative_id_values.append(f"{row_index}:{column}={value_text}")
            if column in {"N", "K", "horizon", "steps", "compute_rollouts", "rollouts", "pool_size"} and parsed <= 0:
                nonpositive_compute_values.append(f"{row_index}:{column}={value_text}")
            if "success" in lower and not any(skip in lower for skip in ("count", "mae", "error")) and not 0.0 <= parsed <= 1.0:
                success_range_errors.append(f"{row_index}:{column}={value_text}")

    duplicate_columns = sorted({column for column in fieldnames if fieldnames.count(column) > 1})
    blank_column_names = [column for column in fieldnames if not str(column).strip()]
    family_failures = []
    field_set = set(fieldnames)
    for check_name, options, detail in expected_family_requirements(path.name):
        if not field_set.intersection(options):
            family_failures.append({"name": check_name, "options": sorted(options), "detail": detail})

    return {
        "path": path.as_posix(),
        "rows": len(rows),
        "columns": len(fieldnames),
        "fieldnames": fieldnames,
        "numeric_columns": sorted(numeric_columns),
        "blank_columns": sorted(blank_columns),
        "malformed": malformed,
        "duplicate_columns": duplicate_columns,
        "blank_column_names": blank_column_names,
        "duplicate_rows": duplicate_rows,
        "blank_disallowed": blank_disallowed,
        "nonfinite_numeric": nonfinite_numeric,
        "noninteger_values": noninteger_values,
        "negative_id_values": negative_id_values,
        "nonpositive_compute_values": nonpositive_compute_values,
        "success_range_errors": success_range_errors,
        "boolean_errors": boolean_errors,
        "family_failures": family_failures,
    }


def audit_table_schemas(root: Path, results_dir: Path | None = None) -> dict[str, Any]:
    root = root.resolve()
    results_dir = (results_dir or root / "results").resolve()
    paths = table_paths(results_dir)
    records = [audit_table(path) for path in paths]
    checks: list[TableSchemaCheck] = []

    total_rows = sum(int(record["rows"]) for record in records)
    total_columns = sum(int(record["columns"]) for record in records)
    numeric_column_instances = sum(len(record["numeric_columns"]) for record in records)
    optional_blank_instances = sum(sum(1 for column in record["blank_columns"] if column in OPTIONAL_BLANK_COLUMNS) for record in records)

    empty_tables = [record["path"] for record in records if int(record["rows"]) <= 0]
    header_issues = [record["path"] for record in records if int(record["columns"]) <= 0 or record["duplicate_columns"] or record["blank_column_names"]]
    malformed_tables = [record["path"] for record in records if record["malformed"]]
    duplicate_row_tables = [record["path"] for record in records if int(record["duplicate_rows"]) > 0]
    blank_issues = [record["path"] for record in records if record["blank_disallowed"]]
    numeric_issues = [record["path"] for record in records if record["nonfinite_numeric"]]
    integer_issues = [record["path"] for record in records if record["noninteger_values"]]
    id_issues = [record["path"] for record in records if record["negative_id_values"]]
    compute_issues = [record["path"] for record in records if record["nonpositive_compute_values"]]
    success_issues = [record["path"] for record in records if record["success_range_errors"]]
    boolean_issues = [record["path"] for record in records if record["boolean_errors"]]
    family_issues = [record["path"] for record in records if record["family_failures"]]

    add(checks, "table_files_present", len(paths) >= 200, f"tables={len(paths)}")
    add(checks, "table_rows_present", total_rows >= 200_000, f"rows={total_rows}")
    add(checks, "table_columns_present", total_columns >= 1_500, f"column_instances={total_columns}")
    add(checks, "table_headers_well_formed", not header_issues, f"tables={header_issues[:10]}, count={len(header_issues)}")
    add(checks, "table_rows_nonempty", not empty_tables, f"tables={empty_tables[:10]}, count={len(empty_tables)}")
    add(checks, "table_rows_not_malformed", not malformed_tables, f"tables={malformed_tables[:10]}, count={len(malformed_tables)}")
    add(checks, "table_rows_unique", not duplicate_row_tables, f"tables={duplicate_row_tables[:10]}, count={len(duplicate_row_tables)}")
    add(checks, "nonoptional_cells_nonblank", not blank_issues, f"tables={blank_issues[:10]}, count={len(blank_issues)}")
    add(checks, "numeric_cells_are_finite", not numeric_issues, f"tables={numeric_issues[:10]}, count={len(numeric_issues)}")
    add(checks, "integer_key_cells_are_integral", not integer_issues, f"tables={integer_issues[:10]}, count={len(integer_issues)}")
    add(checks, "identifier_cells_nonnegative", not id_issues, f"tables={id_issues[:10]}, count={len(id_issues)}")
    add(checks, "compute_count_cells_positive", not compute_issues, f"tables={compute_issues[:10]}, count={len(compute_issues)}")
    add(checks, "success_rate_cells_unit_interval", not success_issues, f"tables={success_issues[:10]}, count={len(success_issues)}")
    add(checks, "boolean_cells_parse", not boolean_issues, f"tables={boolean_issues[:10]}, count={len(boolean_issues)}")
    add(checks, "family_specific_columns_present", not family_issues, f"tables={family_issues[:10]}, count={len(family_issues)}")
    add(checks, "numeric_column_coverage", numeric_column_instances >= 500, f"numeric_column_instances={numeric_column_instances}")
    add(checks, "optional_blank_columns_are_explicit", optional_blank_instances > 0, f"optional_blank_instances={optional_blank_instances}")

    issues = [check for check in checks if not check.ok]
    return {
        "experiment": "table_schema",
        "verified": len(issues) == 0,
        "results_dir": str(results_dir),
        "n_tables": len(paths),
        "total_rows": total_rows,
        "total_column_instances": total_columns,
        "numeric_column_instances": numeric_column_instances,
        "optional_blank_instances": optional_blank_instances,
        "n_checks": len(checks),
        "n_issues": len(issues),
        "checks": [check.__dict__ for check in checks],
        "issues": [check.__dict__ for check in issues],
        "table_records": records,
    }


def table_schema_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Table Schema Report",
        "",
        f"- Verified: {payload.get('verified')}",
        f"- Tables audited: {payload.get('n_tables')}",
        f"- Total rows: {payload.get('total_rows')}",
        f"- Numeric column instances: {payload.get('numeric_column_instances')}",
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
        lines.append("Canonical CSV tables have well-formed headers, nonempty rows, unique rows, finite numeric cells, valid key/count ranges, explicit optional blanks, and family-specific columns for curves, exact-law, rollout, seed-metric, train/validation, and closed-loop tables.")
    lines.append("")
    return "\n".join(lines)
