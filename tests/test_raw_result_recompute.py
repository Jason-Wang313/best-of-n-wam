import csv
import json
from pathlib import Path

from wam_inference_value.evaluation import ci95
from wam_inference_value.raw_result_recompute import audit_raw_result_recompute


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def minimal_project(root: Path, *, bad_aggregate: bool = False, bad_ci: bool = False) -> None:
    table = root / "results" / "tables" / "learned.csv"
    aggregate = root / "results" / "tables" / "multi_aggregate.csv"
    curves = root / "results" / "tables" / "multi_curves.csv"
    exact = root / "results" / "tables" / "exact.csv"
    seed_metrics = root / "results" / "tables" / "seed_metrics.csv"

    learned_rows = [
        {"backend": "a", "N": 1, "success": 0.0, "real_utility": 1.0},
        {"backend": "a", "N": 1, "success": 1.0, "real_utility": 3.0},
    ]
    write_rows(table, learned_rows)
    write_json(
        root / "results" / "learned_wam_vs_analytic_wam.json",
        {
            "artifacts": {"table": str(table)},
            "aggregate": [{"backend": "a", "N": 1, "success": 0.5, "real_utility": 2.0}],
        },
    )

    multi_rows = [
        {"env": "e", "backend": "b", "scorer": "s", "mismatch": "m", "N": 1, "success": 0.0, "real_utility": 2.0},
        {"env": "e", "backend": "b", "scorer": "s", "mismatch": "m", "N": 1, "success": 1.0, "real_utility": 4.0},
    ]
    write_rows(curves, multi_rows)
    write_rows(
        aggregate,
        [{"env": "e", "backend": "b", "scorer": "s", "mismatch": "m", "N": 1, "success": 0.7 if bad_aggregate else 0.5, "real_utility": 3.0}],
    )
    write_json(root / "results" / "multi_env_suite.json", {"artifacts": {"curves": str(curves), "aggregate": str(aggregate)}})

    write_rows(exact, [{"utility_abs_error": 0.1}, {"utility_abs_error": 0.3}])
    metric_values = [0.2, 0.4, 0.6]
    write_rows(seed_metrics, [{"seed": i, "delta": value} for i, value in enumerate(metric_values)])
    summary_ci = ci95(metric_values)
    if bad_ci:
        summary_ci["mean"] = 9.0
    write_json(
        root / "results" / "benchmark_example.json",
        {
            "exact_law_utility_mae": 0.2,
            "exact_path": str(exact),
            "seed_metrics_path": str(seed_metrics),
            "confidence_intervals": {"delta": summary_ci},
        },
    )


def test_raw_result_recompute_accepts_matching_raw_tables(tmp_path: Path) -> None:
    minimal_project(tmp_path)

    audit = audit_raw_result_recompute(tmp_path, tmp_path / "results")

    issue_names = {issue["name"] for issue in audit["issues"]}
    assert "learned_wam_vs_analytic_aggregate_means_match_raw_rows" not in issue_names
    assert "multi_env_aggregate_means_match_raw_rows" not in issue_names
    assert "exact_law_mae_matches_raw_tables" not in issue_names
    assert "seed_metric_cis_match_raw_columns" not in issue_names


def test_raw_result_recompute_flags_aggregate_mismatch(tmp_path: Path) -> None:
    minimal_project(tmp_path, bad_aggregate=True)

    audit = audit_raw_result_recompute(tmp_path, tmp_path / "results")

    issue_names = {issue["name"] for issue in audit["issues"]}
    assert "multi_env_aggregate_means_match_raw_rows" in issue_names


def test_raw_result_recompute_flags_seed_ci_mismatch(tmp_path: Path) -> None:
    minimal_project(tmp_path, bad_ci=True)

    audit = audit_raw_result_recompute(tmp_path, tmp_path / "results")

    issue_names = {issue["name"] for issue in audit["issues"]}
    assert "seed_metric_cis_match_raw_columns" in issue_names
