from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


FALLBACK_PYTEST_PASSED = 95


@dataclass(frozen=True)
class CommandResultCheck:
    name: str
    ok: bool
    detail: str


def add(checks: list[CommandResultCheck], name: str, ok: bool, detail: str) -> None:
    checks.append(CommandResultCheck(name=name, ok=bool(ok), detail=detail))


def load_json(results_dir: Path, name: str) -> dict[str, Any]:
    path = results_dir / name
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def command_section(text: str) -> str:
    marker = "## Command Results"
    index = text.find(marker)
    return text[index:] if index >= 0 else ""


def expected_snippets(results_dir: Path) -> list[tuple[str, str]]:
    test_inventory = load_json(results_dir, "test_inventory.json")
    artifact_integrity = load_json(results_dir, "artifact_integrity.json")
    artifact_manifest = load_json(results_dir, "artifact_manifest.json")
    figure_quality = load_json(results_dir, "figure_quality.json")
    result_consistency = load_json(results_dir, "result_consistency.json")
    raw_result_recompute = load_json(results_dir, "raw_result_recompute.json")
    table_schema = load_json(results_dir, "table_schema.json")
    source_manifest = load_json(results_dir, "source_manifest.json")
    runtime_environment = load_json(results_dir, "runtime_environment.json")
    experiment_registry = load_json(results_dir, "experiment_registry.json")
    model_artifact_integrity = load_json(results_dir, "model_artifact_integrity.json")
    narrative_consistency = load_json(results_dir, "narrative_consistency.json")
    script_contracts = load_json(results_dir, "script_contracts.json")
    abstract_claim_support = load_json(results_dir, "abstract_claim_support.json")
    publication_scope = load_json(results_dir, "publication_scope.json")
    frontier_integrity = load_json(results_dir, "frontier_integrity.json")
    ideal_claim_boundary = load_json(results_dir, "ideal_claim_boundary.json")
    claim_semantics = load_json(results_dir, "claim_semantics.json")
    claim_scope_audit = load_json(results_dir, "claim_scope_audit.json")
    claim_reference_integrity = load_json(results_dir, "claim_reference_integrity.json")
    claim_evidence_quality = load_json(results_dir, "claim_evidence_quality.json")
    tracked_artifact_provenance = load_json(results_dir, "tracked_artifact_provenance.json")
    evidence_hash_coverage = load_json(results_dir, "evidence_hash_coverage.json")
    repo_bound_artifact_audit = load_json(results_dir, "repo_bound_artifact_audit.json")
    claim_ledger_integrity = load_json(results_dir, "claim_ledger_integrity.json")
    claim_generation_consistency = load_json(results_dir, "claim_generation_consistency.json")
    report_generation_consistency = load_json(results_dir, "report_generation_consistency.json")
    expected_pytest_passed = test_inventory.get("n_tests") or FALLBACK_PYTEST_PASSED

    runtime_python = (runtime_environment.get("python") or {}).get("version")
    return [
        ("pytest", f"`python -m pytest -q`: passed with `{expected_pytest_passed} passed`"),
        (
            "test_inventory",
            (
                f"`python scripts/test_inventory.py --fail-on-error`: passed with "
                f"`{test_inventory.get('n_tests')}` collected tests, `{test_inventory.get('n_checks')}` inventory checks, "
                f"and `{test_inventory.get('n_issues')}` issues"
            ),
        ),
        (
            "artifact_integrity",
            (
                f"`python scripts/artifact_integrity.py --fail-on-error`: passed with "
                f"`{artifact_integrity.get('n_references')}` artifact references checked and "
                f"`{artifact_integrity.get('n_issues')}` issues"
            ),
        ),
        (
            "artifact_manifest",
            (
                f"`python scripts/artifact_manifest.py --fail-on-error`: passed with "
                f"`{artifact_manifest.get('n_files')}` scientific artifacts, "
                f"`{artifact_manifest.get('total_bytes')}` bytes, "
                f"`{artifact_manifest.get('n_checks')}` manifest checks, and "
                f"`{artifact_manifest.get('n_issues')}` issues"
            ),
        ),
        (
            "figure_quality",
            (
                f"`python scripts/figure_quality.py --fail-on-error`: passed with "
                f"`{figure_quality.get('n_figures')}` figures, "
                f"`{figure_quality.get('n_checks')}` image-quality checks, and "
                f"`{figure_quality.get('n_issues')}` issues"
            ),
        ),
        (
            "result_consistency",
            (
                f"`python scripts/result_consistency.py --fail-on-error`: passed with "
                f"`{result_consistency.get('n_checks')}` consistency checks and "
                f"`{result_consistency.get('n_issues')}` issues"
            ),
        ),
        (
            "raw_result_recompute",
            (
                f"`python scripts/raw_result_recompute.py --fail-on-error`: passed with "
                f"`{raw_result_recompute.get('aggregate_metrics_compared')}` aggregate metrics, "
                f"`{raw_result_recompute.get('exact_law_mae_files')}` exact-law files, "
                f"`{raw_result_recompute.get('seed_metric_ci_columns')}` seed CI columns, and "
                f"`{raw_result_recompute.get('n_issues')}` issues"
            ),
        ),
        (
            "table_schema",
            (
                f"`python scripts/table_schema.py --fail-on-error`: passed with "
                f"`{table_schema.get('n_tables')}` tables, `{table_schema.get('total_rows')}` rows, "
                f"`{table_schema.get('n_checks')}` schema checks, and `{table_schema.get('n_issues')}` issues"
            ),
        ),
        (
            "source_manifest",
            (
                f"`python scripts/source_manifest.py --fail-on-error`: passed with "
                f"`{source_manifest.get('n_files')}` source files, `{source_manifest.get('total_bytes')}` bytes, "
                f"`{source_manifest.get('n_checks')}` source-manifest checks, and `{source_manifest.get('n_issues')}` issues"
            ),
        ),
        (
            "runtime_environment",
            (
                f"`python scripts/runtime_environment.py --fail-on-error`: passed with Python `{runtime_python}`, "
                f"`{runtime_environment.get('n_core_requirements')}` core requirements, "
                f"`{runtime_environment.get('n_optional_available')}` / "
                f"`{runtime_environment.get('n_optional_requirements')}` optional requirements available, "
                f"`{runtime_environment.get('n_checks')}` runtime checks, and "
                f"`{runtime_environment.get('n_issues')}` issues"
            ),
        ),
        (
            "experiment_registry",
            (
                f"`python scripts/experiment_registry.py --fail-on-error`: passed with "
                f"`{experiment_registry.get('n_entries')}` experiment-family entries, "
                f"`{experiment_registry.get('n_wrapper_links')}` wrapper links, "
                f"`{experiment_registry.get('n_table_artifacts')}` table artifacts, "
                f"`{experiment_registry.get('n_figure_artifacts')}` figures, "
                f"`{experiment_registry.get('n_checks')}` registry checks, and "
                f"`{experiment_registry.get('n_issues')}` issues"
            ),
        ),
        (
            "model_artifact_integrity",
            (
                f"`python scripts/model_artifact_integrity.py --fail-on-error`: passed with "
                f"`{model_artifact_integrity.get('n_models')}` model artifacts, "
                f"`{model_artifact_integrity.get('n_npz_arrays')}` NPZ arrays, "
                f"`{model_artifact_integrity.get('n_joblib_predictors')}` joblib predictors, "
                f"`{model_artifact_integrity.get('n_checks')}` model-artifact checks, and "
                f"`{model_artifact_integrity.get('n_issues')}` issues"
            ),
        ),
        (
            "narrative_consistency",
            (
                f"`python scripts/narrative_consistency.py --fail-on-error`: passed with "
                f"`{narrative_consistency.get('n_checks')}` narrative checks and "
                f"`{narrative_consistency.get('n_issues')}` issues"
            ),
        ),
        (
            "script_contracts",
            (
                f"`python scripts/script_contracts.py --fail-on-error`: passed with "
                f"`{script_contracts.get('n_scripts')}` scripts, `{script_contracts.get('n_checks')}` contract checks, "
                f"and `{script_contracts.get('n_issues')}` issues"
            ),
        ),
        (
            "abstract_claim_support",
            (
                f"`python scripts/abstract_claim_support.py --fail-on-error`: passed with "
                f"`{abstract_claim_support.get('n_abstract_claims')}` abstract claims, "
                f"`{abstract_claim_support.get('n_backing_claim_links')}` backing claim links, "
                f"`{abstract_claim_support.get('n_forbidden_headline_hits')}` forbidden headline hits, and "
                f"`{abstract_claim_support.get('n_issues')}` issues"
            ),
        ),
        (
            "publication_scope",
            (
                f"`python scripts/publication_scope.py --fail-on-error`: passed with "
                f"`{publication_scope.get('n_publication_surfaces')}` publication surfaces, "
                f"`{publication_scope.get('n_risk_mentions')}` risky mentions, "
                f"`{publication_scope.get('n_unguarded_mentions')}` unguarded mentions, and "
                f"`{publication_scope.get('n_issues')}` issues"
            ),
        ),
        (
            "frontier_integrity",
            (
                f"`python scripts/frontier_integrity.py --fail-on-error`: passed with "
                f"`{frontier_integrity.get('n_frontier_items')}` frontier items, "
                f"`{frontier_integrity.get('n_guarded_frontier_mentions')}` guarded mentions, "
                f"`{frontier_integrity.get('n_promoted_frontier_claims')}` promoted frontier claims, and "
                f"`{frontier_integrity.get('n_issues')}` issues"
            ),
        ),
        (
            "ideal_claim_boundary",
            (
                f"`python scripts/ideal_claim_boundary.py --fail-on-error`: passed with "
                f"`{ideal_claim_boundary.get('n_ideal_claims')}` ideal claims, "
                f"`{ideal_claim_boundary.get('n_promotable_claims')}` promotable claims, "
                f"`{ideal_claim_boundary.get('n_future_only_claims')}` future-only claims, "
                f"`{ideal_claim_boundary.get('all_ideal_claims_promotable')}` all-promotable flag, and "
                f"`{ideal_claim_boundary.get('n_issues')}` issues"
            ),
        ),
        (
            "claim_semantics",
            (
                f"`python scripts/claim_semantics.py --fail-on-error`: passed with "
                f"`{claim_semantics.get('n_claims')}` claims, `{claim_semantics.get('n_checks')}` semantic checks, "
                f"`{claim_semantics.get('n_ci_claims')}` CI-backed claims, and `{claim_semantics.get('n_issues')}` issues"
            ),
        ),
        (
            "claim_scope_audit",
            (
                f"`python scripts/claim_scope_audit.py --fail-on-error`: passed with "
                f"`{claim_scope_audit.get('n_claims')}` claims, `{claim_scope_audit.get('n_scope_mentions')}` scoped broad-claim mentions, "
                f"`{claim_scope_audit.get('n_checks')}` checks, and `{claim_scope_audit.get('n_issues')}` issues"
            ),
        ),
        (
            "claim_reference_integrity",
            (
                f"`python scripts/claim_reference_integrity.py --fail-on-error`: passed with "
                f"`{claim_reference_integrity.get('n_references')}` explicit verified-claim references, "
                f"`{claim_reference_integrity.get('n_unique_referenced_claims')}` unique referenced claims, "
                f"and `{claim_reference_integrity.get('n_issues')}` issues"
            ),
        ),
        (
            "claim_evidence_quality",
            (
                f"`python scripts/claim_evidence_quality.py --fail-on-error`: passed with "
                f"`{claim_evidence_quality.get('n_claims')}` claims, `{claim_evidence_quality.get('n_source_links')}` source links, "
                f"`{claim_evidence_quality.get('n_checks')}` evidence checks, and `{claim_evidence_quality.get('n_issues')}` issues"
            ),
        ),
        (
            "tracked_artifact_provenance",
            (
                f"`python scripts/tracked_artifact_provenance.py --fail-on-error`: passed with "
                f"`{tracked_artifact_provenance.get('n_claim_sources')}` claim sources, "
                f"`{tracked_artifact_provenance.get('n_artifact_references')}` artifact references, and "
                f"`{tracked_artifact_provenance.get('n_issues')}` issues"
            ),
        ),
        (
            "repo_bound_artifact_audit",
            (
                f"`python scripts/repo_bound_artifact_audit.py --fail-on-error`: passed with "
                f"`{repo_bound_artifact_audit.get('n_records')}` records, "
                f"`{repo_bound_artifact_audit.get('n_claim_sources')}` claim sources, "
                f"`{repo_bound_artifact_audit.get('n_artifact_references')}` artifact references, "
                f"`{repo_bound_artifact_audit.get('n_outside_repo')}` outside-repo records, "
                f"`{repo_bound_artifact_audit.get('n_missing')}` missing records, and "
                f"`{repo_bound_artifact_audit.get('n_issues')}` issues"
            ),
        ),
        (
            "evidence_hash_coverage",
            (
                f"`python scripts/evidence_hash_coverage.py --fail-on-error`: passed with "
                f"`{evidence_hash_coverage.get('n_claim_sources')}` claim sources, "
                f"`{evidence_hash_coverage.get('n_artifact_references')}` artifact references, "
                f"`{evidence_hash_coverage.get('n_hashed_records')}` hashed records, and "
                f"`{evidence_hash_coverage.get('n_issues')}` issues"
            ),
        ),
        (
            "claim_ledger_integrity",
            (
                f"`python scripts/claim_ledger_integrity.py --fail-on-error`: passed with "
                f"`{claim_ledger_integrity.get('n_claims')}` claims, `{claim_ledger_integrity.get('n_checks')}` ledger checks, "
                f"and `{claim_ledger_integrity.get('n_issues')}` issues"
            ),
        ),
        (
            "claim_generation_consistency",
            (
                f"`python scripts/claim_generation_consistency.py --fail-on-error`: passed with "
                f"`{claim_generation_consistency.get('n_claims')}` claims, "
                f"`{claim_generation_consistency.get('n_checks')}` generation checks, and "
                f"`{claim_generation_consistency.get('n_issues')}` issues"
            ),
        ),
        (
            "report_generation_consistency",
            (
                f"`python scripts/report_generation_consistency.py --fail-on-error`: passed with "
                f"`{report_generation_consistency.get('n_reports')}` reports, "
                f"`{report_generation_consistency.get('n_checks')}` generation checks, and "
                f"`{report_generation_consistency.get('n_issues')}` issues"
            ),
        ),
    ]


def audit_command_result_consistency(root: Path, results_dir: Path | None = None) -> dict[str, Any]:
    root = root.resolve()
    results_dir = (results_dir or root / "results").resolve()
    text = read_text(root / "reports" / "final_decision_report.md")
    section = command_section(text)
    checks: list[CommandResultCheck] = []
    snippets = expected_snippets(results_dir)

    add(checks, "command_section_present", bool(section), "final_decision_report.md has ## Command Results")
    for name, snippet in snippets:
        add(checks, f"{name}_snippet_matches_artifact", snippet in section, f"expected={snippet}")

    python_command_lines = re.findall(r"^- `python [^`]+`:", section, flags=re.MULTILINE)
    add(checks, "python_command_lines_present", len(python_command_lines) >= len(snippets) - 1, f"lines={len(python_command_lines)}")
    stale_tokens = [
        "`71 passed`",
        "claim_ledger_integrity.py --fail-on-error`: passed with `114` claims, `31` ledger checks, and `1` issues",
    ]
    stale_found = [token for token in stale_tokens if token in section]
    add(checks, "no_known_stale_command_tokens", not stale_found, f"found={stale_found}")

    issues = [check for check in checks if not check.ok]
    return {
        "experiment": "command_result_consistency",
        "verified": len(issues) == 0,
        "n_expected_snippets": len(snippets),
        "n_python_command_lines": len(python_command_lines),
        "n_checks": len(checks),
        "n_issues": len(issues),
        "expected_pytest_passed": (load_json(results_dir, "test_inventory.json").get("n_tests") or FALLBACK_PYTEST_PASSED),
        "checks": [check.__dict__ for check in checks],
        "issues": [check.__dict__ for check in issues],
    }


def command_result_consistency_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Command Result Consistency Report",
        "",
        f"- Verified: {payload.get('verified')}",
        f"- Expected snippets: {payload.get('n_expected_snippets')}",
        f"- Python command lines: {payload.get('n_python_command_lines')}",
        f"- Checks: {payload.get('n_checks')}",
        f"- Issues: {payload.get('n_issues')}",
        f"- Expected pytest count: {payload.get('expected_pytest_passed')}",
        "",
    ]
    issues = payload.get("issues") or []
    if issues:
        lines.append("## Issues")
        lines.append("")
        for issue in issues:
            lines.append(f"- `{issue.get('name')}`: {issue.get('detail')}")
    else:
        lines.append("The final decision command-results section matches current verification artifacts and contains no known stale command tokens.")
    lines.append("")
    return "\n".join(lines)
