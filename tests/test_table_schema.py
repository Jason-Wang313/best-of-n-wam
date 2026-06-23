from pathlib import Path

from wam_inference_value.table_schema import audit_table_schemas


def write_table(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_table_schema_accepts_valid_curve_table(tmp_path: Path) -> None:
    results = tmp_path / "results"
    write_table(
        results / "tables" / "demo_curves.csv",
        "seed,state_id,scorer,N,real_utility,success\n"
        "0,0,random,1,-1.0,0.0\n"
        "0,0,oracle,2,1.0,1.0\n",
    )

    audit = audit_table_schemas(tmp_path, results)

    issue_names = {issue["name"] for issue in audit["issues"]}
    assert "numeric_cells_are_finite" not in issue_names
    assert "family_specific_columns_present" not in issue_names


def test_table_schema_detects_nonfinite_numeric_cell(tmp_path: Path) -> None:
    results = tmp_path / "results"
    write_table(
        results / "tables" / "demo_curves.csv",
        "seed,state_id,scorer,N,real_utility,success\n"
        "0,0,random,1,nan,0.0\n",
    )

    audit = audit_table_schemas(tmp_path, results)

    assert "numeric_cells_are_finite" in {issue["name"] for issue in audit["issues"]}


def test_table_schema_detects_missing_curve_metric(tmp_path: Path) -> None:
    results = tmp_path / "results"
    write_table(
        results / "tables" / "demo_curves.csv",
        "seed,state_id,scorer,N\n"
        "0,0,random,1\n",
    )

    audit = audit_table_schemas(tmp_path, results)

    assert "family_specific_columns_present" in {issue["name"] for issue in audit["issues"]}


def test_table_schema_detects_duplicate_rows(tmp_path: Path) -> None:
    results = tmp_path / "results"
    write_table(
        results / "tables" / "demo_exact_law.csv",
        "seed,state_id,N,utility_abs_error\n"
        "0,0,1,0.1\n"
        "0,0,1,0.1\n",
    )

    audit = audit_table_schemas(tmp_path, results)

    assert "table_rows_unique" in {issue["name"] for issue in audit["issues"]}


def test_table_schema_allows_blank_step_error(tmp_path: Path) -> None:
    results = tmp_path / "results"
    write_table(
        results / "tables" / "modern_vla_libero_policy_eval_episodes.csv",
        "episode_id,seed,steps,success,step_error\n"
        "0,300,1,False,\n",
    )

    audit = audit_table_schemas(tmp_path, results)

    assert "nonoptional_cells_nonblank" not in {issue["name"] for issue in audit["issues"]}
