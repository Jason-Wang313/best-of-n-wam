import json
import re
from pathlib import Path

from wam_inference_value.command_result_consistency import audit_command_result_consistency, expected_snippets


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_minimal_gate_artifacts(results: Path) -> None:
    write_json(results / "test_inventory.json", {"n_tests": 95, "n_checks": 6, "n_issues": 0})
    write_json(results / "artifact_integrity.json", {"n_references": 641, "n_issues": 0})
    write_json(results / "artifact_manifest.json", {"n_files": 397, "total_bytes": 56, "n_checks": 15, "n_issues": 0})
    write_json(results / "figure_quality.json", {"n_figures": 36, "n_checks": 8, "n_issues": 0})
    write_json(results / "result_consistency.json", {"n_checks": 157, "n_issues": 0})
    write_json(
        results / "raw_result_recompute.json",
        {"aggregate_metrics_compared": 10710, "exact_law_mae_files": 20, "seed_metric_ci_columns": 125, "n_issues": 0},
    )
    write_json(results / "table_schema.json", {"n_tables": 215, "total_rows": 212858, "n_checks": 17, "n_issues": 0})
    write_json(results / "source_manifest.json", {"n_files": 170, "total_bytes": 123, "n_checks": 14, "n_issues": 0})
    write_json(
        results / "runtime_environment.json",
        {"python": {"version": "3.10.11"}, "n_core_requirements": 4, "n_optional_available": 5, "n_optional_requirements": 5, "n_checks": 15, "n_issues": 0},
    )
    write_json(
        results / "experiment_registry.json",
        {"n_entries": 57, "n_wrapper_links": 69, "n_table_artifacts": 297, "n_figure_artifacts": 40, "n_checks": 10, "n_issues": 0},
    )
    write_json(
        results / "model_artifact_integrity.json",
        {"n_models": 49, "n_npz_arrays": 315, "n_joblib_predictors": 13, "n_checks": 10, "n_issues": 0},
    )
    write_json(results / "narrative_consistency.json", {"n_checks": 40, "n_issues": 0})
    write_json(results / "script_contracts.json", {"n_scripts": 7, "n_checks": 77, "n_issues": 0})
    write_json(results / "abstract_claim_support.json", {"n_abstract_claims": 4, "n_backing_claim_links": 23, "n_forbidden_headline_hits": 0, "n_issues": 0})
    write_json(results / "publication_scope.json", {"n_publication_surfaces": 5, "n_risk_mentions": 72, "n_unguarded_mentions": 0, "n_issues": 0})
    write_json(results / "claim_semantics.json", {"n_claims": 114, "n_checks": 168, "n_ci_claims": 51, "n_issues": 0})
    write_json(results / "claim_evidence_quality.json", {"n_claims": 114, "n_source_links": 149, "n_checks": 7, "n_issues": 0})
    write_json(results / "tracked_artifact_provenance.json", {"n_claim_sources": 149, "n_artifact_references": 641, "n_issues": 0})
    write_json(results / "repo_bound_artifact_audit.json", {"n_records": 790, "n_claim_sources": 149, "n_artifact_references": 641, "n_outside_repo": 0, "n_missing": 0, "n_issues": 0})
    write_json(results / "evidence_hash_coverage.json", {"n_claim_sources": 149, "n_artifact_references": 641, "n_hashed_records": 790, "n_issues": 0})
    write_json(results / "claim_ledger_integrity.json", {"n_claims": 114, "n_checks": 31, "n_issues": 0})
    write_json(results / "claim_generation_consistency.json", {"n_claims": 114, "n_checks": 10, "n_issues": 0})
    write_json(results / "report_generation_consistency.json", {"n_reports": 8, "n_checks": 9, "n_issues": 0})


def write_final_report(root: Path, results: Path, *, stale_pytest: bool = False, stale_ledger: bool = False) -> None:
    named_snippets = expected_snippets(results)
    snippets = [snippet for _, snippet in named_snippets]
    if stale_pytest:
        snippets[0] = re.sub(r"\d+ passed", "71 passed", snippets[0])
    if stale_ledger:
        ledger_index = next(index for index, (name, _) in enumerate(named_snippets) if name == "claim_ledger_integrity")
        snippets[ledger_index] = snippets[ledger_index].replace("and `0` issues", "and `1` issues")
    report = root / "reports" / "final_decision_report.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("## Command Results\n\n" + "\n".join(f"- {snippet}." for snippet in snippets), encoding="utf-8")


def test_command_result_consistency_accepts_current_report(tmp_path: Path):
    results = tmp_path / "results"
    write_minimal_gate_artifacts(results)
    write_final_report(tmp_path, results)

    payload = audit_command_result_consistency(tmp_path, results)

    assert payload["verified"] is True
    assert payload["n_issues"] == 0


def test_command_result_consistency_rejects_stale_pytest_count(tmp_path: Path):
    results = tmp_path / "results"
    write_minimal_gate_artifacts(results)
    write_final_report(tmp_path, results, stale_pytest=True)

    payload = audit_command_result_consistency(tmp_path, results)

    failures = {check["name"] for check in payload["issues"]}
    assert "pytest_snippet_matches_artifact" in failures
    assert "no_known_stale_command_tokens" in failures


def test_command_result_consistency_rejects_stale_ledger_issue_count(tmp_path: Path):
    results = tmp_path / "results"
    write_minimal_gate_artifacts(results)
    write_final_report(tmp_path, results, stale_ledger=True)

    payload = audit_command_result_consistency(tmp_path, results)

    failures = {check["name"] for check in payload["issues"]}
    assert "claim_ledger_integrity_snippet_matches_artifact" in failures
    assert "no_known_stale_command_tokens" in failures
