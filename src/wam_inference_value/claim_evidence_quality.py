from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PLACEHOLDER_RE = re.compile(r"\b(?:None|NaN|nan|TBD|TODO|pending|missing)\b")
STRUCTURED_EVIDENCE_RE = re.compile(r"\d|=|\{|\[|:")


@dataclass(frozen=True)
class ClaimEvidenceCheck:
    name: str
    ok: bool
    detail: str


def add(checks: list[ClaimEvidenceCheck], name: str, ok: bool, detail: str) -> None:
    checks.append(ClaimEvidenceCheck(name=name, ok=bool(ok), detail=detail))


def set_sources(mapping: dict[int, list[str]], ids: list[int] | range, sources: list[str]) -> None:
    for cid in ids:
        mapping[int(cid)] = list(sources)


def default_claim_source_map() -> dict[int, list[str]]:
    mapping: dict[int, list[str]] = {}
    set_sources(mapping, [1, 2], ["results/exp1_exact_rollout_law_validation.json"])
    set_sources(mapping, [3, 4], ["results/exp2_auc_vs_moment_hierarchy.json"])
    set_sources(mapping, [5, 6], ["results/exp3_pilot_to_heldout_prediction.json"])
    set_sources(mapping, [7], ["results/exp4_score_function_comparison.json"])
    set_sources(mapping, [8], ["results/exp4_score_function_comparison_learned.json"])
    set_sources(mapping, [9, 25], ["results/learned_wam_vs_analytic_wam.json"])
    set_sources(mapping, [10], ["results/exp5_real_vs_imagined_utility_gap.json"])
    set_sources(mapping, [11], ["results/exp5_real_vs_imagined_utility_gap_learned.json"])
    set_sources(mapping, [12, 13], ["results/exp10_falsification_bad_scorer.json", "results/multi_env_suite.json"])
    set_sources(mapping, [14, 15], ["results/exp6_adaptive_rollout_allocation_learned.json"])
    set_sources(mapping, [16], ["results/exp7_closed_loop_receding_horizon_eval.json", "results/exp7_closed_loop_receding_horizon_eval_learned.json"])
    set_sources(mapping, [17, 18], ["results/exp7_closed_loop_receding_horizon_eval_learned.json"])
    set_sources(mapping, [19, 20, 21], ["results/exp8_nonstationary_dynamics_extension.json"])
    set_sources(mapping, [22, 23, 24], ["results/learned_wam_lite_training.json"])
    set_sources(mapping, range(26, 31), ["results/multi_env_suite.json", "results/tables/multi_env_curves.csv", "results/tables/maxout_model_metrics.csv"])
    set_sources(mapping, [31], ["results/benchmark_smoke.json", "src/wam_inference_value/benchmarks/base.py", "src/wam_inference_value/benchmarks/maniskill_adapter.py"])
    set_sources(mapping, [32], ["results/benchmark_rollout_pools.json"])
    set_sources(mapping, [33], ["results/benchmark_exact_law_validation.json"])
    set_sources(mapping, [34], ["results/benchmark_score_comparison.json"])
    set_sources(mapping, [35], ["results/benchmark_real_vs_imagined_gap.json"])
    set_sources(mapping, [36], ["results/benchmark_closed_loop_eval.json"])
    set_sources(mapping, [37], ["results/benchmark_wam_training.json"])
    set_sources(mapping, [38, 39], ["results/visual_optional.json"])
    set_sources(mapping, [40], ["results/benchmark_visual_optional.json"])
    set_sources(mapping, range(41, 45), ["results/inference_audit_framework.json"])
    set_sources(mapping, [45], ["results/inference_audit_framework_learned.json"])
    set_sources(mapping, [46], ["results/scorer_repair_experiment.json"])
    set_sources(mapping, [47], ["results/imagination_scaling_law.json"])
    set_sources(mapping, range(48, 55), ["results/benchmark_maniskill_suite.json"])
    set_sources(mapping, range(55, 59), ["results/benchmark_visual_wam_lite.json"])
    set_sources(mapping, range(59, 64), ["results/benchmark_gym_robotics_suite.json"])
    set_sources(mapping, range(64, 68), ["results/benchmark_gym_robotics_visual_wam.json"])
    set_sources(mapping, [68], ["results/benchmark_maniskill_visual_probe.json", "results/benchmark_maniskill_dependency_probe.json"])
    set_sources(mapping, range(69, 73), ["results/benchmark_metaworld_suite.json"])
    set_sources(mapping, range(73, 78), ["results/benchmark_robosuite_suite.json"])
    set_sources(mapping, [78], ["results/benchmark_robocasa_smoke.json"])
    set_sources(mapping, [79], ["results/benchmark_robocasa_learned_wam.json"])
    set_sources(mapping, [80], ["results/benchmark_robocasa_multitask_wam.json"])
    set_sources(mapping, [81], ["README.md", "results/claims_status.json"])
    set_sources(mapping, [82], ["paper_outline.md", "results/claims_status.json"])
    set_sources(mapping, [83], ["results/benchmark_libero_wam.json"])
    set_sources(mapping, [84], ["results/benchmark_robocasa_broad_wam.json"])
    set_sources(mapping, [85], ["results/benchmark_robocasa_family12_wam.json"])
    set_sources(mapping, [86], ["results/benchmark_libero_scripted_policy.json"])
    set_sources(mapping, [87], ["results/benchmark_libero_learned_action_head.json"])
    set_sources(mapping, [88], ["results/benchmark_libero_autonomous_bc_policy.json"])
    set_sources(mapping, [89], ["results/benchmark_libero_visual_language_bc_policy.json"])
    set_sources(mapping, [90], ["results/benchmark_robocasa_family24_wam.json"])
    set_sources(mapping, [91], ["results/benchmark_robocasa_catalog_probe.json"])
    set_sources(mapping, [92], ["results/benchmark_robocasa_micro_rollout_extra.json"])
    set_sources(mapping, [93], ["results/benchmark_robocasa_extra4_wam.json"])
    set_sources(mapping, [94], ["results/benchmark_robocasa_family28_wam.json"])
    set_sources(mapping, [95], ["results/benchmark_robocasa_family32_wam.json"])
    set_sources(mapping, [96], ["results/benchmark_robocasa_stratified55_wam.json"])
    set_sources(mapping, [97], ["results/benchmark_robocasa_stratified97_wam.json"])
    set_sources(mapping, [98], ["results/benchmark_robocasa_residual35_h1_n4_wam.json"])
    set_sources(mapping, [99], ["results/claims_status.json", "reports/claims_report.md"])
    set_sources(mapping, [100], ["results/artifact_integrity.json", "reports/artifact_integrity_report.md"])
    set_sources(mapping, [101], ["results/result_consistency.json", "reports/result_consistency_report.md"])
    set_sources(mapping, [102], ["results/narrative_consistency.json", "reports/narrative_consistency_report.md"])
    set_sources(mapping, [103], ["results/claim_ledger_integrity.json", "reports/claim_ledger_integrity_report.md", "results/claims_status.json"])
    set_sources(mapping, [104], ["results/script_contracts.json", "reports/script_contracts_report.md"])
    set_sources(mapping, [105], ["results/claim_evidence_quality.json", "reports/claim_evidence_quality_report.md"])
    set_sources(mapping, [106], ["results/raw_result_recompute.json", "reports/raw_result_recompute_report.md"])
    set_sources(mapping, [107], ["results/claim_semantics.json", "reports/claim_semantics_report.md"])
    set_sources(mapping, [108], ["results/artifact_manifest.json", "reports/artifact_manifest_report.md"])
    set_sources(mapping, [109], ["results/figure_quality.json", "reports/figure_quality_report.md"])
    set_sources(mapping, [110], ["results/table_schema.json", "reports/table_schema_report.md"])
    set_sources(mapping, [111], ["results/source_manifest.json", "reports/source_manifest_report.md"])
    set_sources(mapping, [112], ["results/runtime_environment.json", "reports/runtime_environment_report.md"])
    return mapping


CLAIM_SOURCE_MAP = default_claim_source_map()


def resolve_path(root: Path, raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    return path if path.is_absolute() else root / path


def _csv_rows(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        rows = list(reader)
    return max(0, len(rows) - 1)


def validate_source(root: Path, raw_path: str) -> dict[str, Any]:
    path = resolve_path(root, raw_path)
    record: dict[str, Any] = {"raw_path": raw_path, "resolved_path": str(path), "exists": path.exists(), "ok": False}
    if not path.exists():
        record["detail"] = "missing"
        return record
    if path.is_dir():
        record["ok"] = any(path.iterdir())
        record["detail"] = "directory"
        return record
    record["bytes"] = path.stat().st_size
    if path.stat().st_size <= 0:
        record["detail"] = "empty"
        return record
    suffix = path.suffix.lower()
    try:
        if suffix == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            record["ok"] = bool(data)
            record["detail"] = "json"
        elif suffix == ".csv":
            rows = _csv_rows(path)
            record["rows"] = rows
            record["ok"] = rows > 0
            record["detail"] = "csv"
        else:
            text = path.read_text(encoding="utf-8", errors="ignore")
            record["ok"] = bool(text.strip())
            record["detail"] = suffix.lstrip(".") or "file"
    except Exception as exc:  # pragma: no cover - defensive, surfaced in JSON.
        record["detail"] = f"parse_error={type(exc).__name__}: {exc}"
        record["ok"] = False
    return record


def audit_claim_evidence_payload(
    payload: dict[str, Any],
    *,
    root: Path,
    source_map: dict[int, list[str]] | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    source_map = CLAIM_SOURCE_MAP if source_map is None else source_map
    claims = [claim for claim in (payload.get("claims") or []) if isinstance(claim, dict)]
    checks: list[ClaimEvidenceCheck] = []
    claim_ids = sorted(int(claim.get("id")) for claim in claims if isinstance(claim.get("id"), int))
    mapped_ids = sorted(source_map)
    missing_source_map_ids = [cid for cid in claim_ids if cid not in source_map]
    extra_source_map_ids = [cid for cid in mapped_ids if cid <= max(claim_ids or [0]) and cid not in claim_ids]

    add(checks, "claims_present", bool(claims), f"claims={len(claims)}")
    add(checks, "all_current_claims_have_sources", not missing_source_map_ids, f"missing={missing_source_map_ids}")
    add(checks, "source_map_has_no_current_extras", not extra_source_map_ids, f"extras={extra_source_map_ids}")

    evidence_placeholder_ids = []
    weak_evidence_ids = []
    ci_claims_without_ci_evidence = []
    source_records = []
    missing_or_invalid_sources = []
    for claim in claims:
        cid = int(claim.get("id"))
        evidence = str(claim.get("evidence") or "")
        claim_text = str(claim.get("claim") or "")
        if PLACEHOLDER_RE.search(evidence):
            evidence_placeholder_ids.append(cid)
        if claim.get("status") == "VERIFIED" and not STRUCTURED_EVIDENCE_RE.search(evidence):
            weak_evidence_ids.append(cid)
        if "CI" in claim_text and "CI" not in evidence and "ci" not in evidence:
            ci_claims_without_ci_evidence.append(cid)
        for raw_path in source_map.get(cid, []):
            record = {"claim_id": cid, **validate_source(root, raw_path)}
            source_records.append(record)
            if not record.get("ok"):
                missing_or_invalid_sources.append(record)

    add(checks, "evidence_has_no_placeholder_literals", not evidence_placeholder_ids, f"claim_ids={evidence_placeholder_ids}")
    add(checks, "verified_evidence_is_structured", not weak_evidence_ids, f"claim_ids={weak_evidence_ids}")
    add(checks, "ci_claims_have_ci_evidence", not ci_claims_without_ci_evidence, f"claim_ids={ci_claims_without_ci_evidence}")
    add(checks, "mapped_sources_exist_and_parse", not missing_or_invalid_sources, f"sources={len(source_records)}, invalid={len(missing_or_invalid_sources)}")

    issues = [check for check in checks if not check.ok]
    return {
        "experiment": "claim_evidence_quality",
        "verified": len(issues) == 0,
        "n_claims": len(claims),
        "max_claim_id": max(claim_ids or [0]),
        "n_source_mapped_claims": len([cid for cid in claim_ids if cid in source_map]),
        "n_source_links": len(source_records),
        "n_checks": len(checks),
        "n_issues": len(issues),
        "checks": [check.__dict__ for check in checks],
        "issues": [check.__dict__ for check in issues],
        "missing_source_map_ids": missing_source_map_ids,
        "extra_source_map_ids": extra_source_map_ids,
        "evidence_placeholder_ids": evidence_placeholder_ids,
        "weak_evidence_ids": weak_evidence_ids,
        "ci_claims_without_ci_evidence": ci_claims_without_ci_evidence,
        "missing_or_invalid_sources": missing_or_invalid_sources,
        "source_records": source_records,
    }


def audit_claim_evidence_quality(root: Path, results_dir: Path | None = None) -> dict[str, Any]:
    root = root.resolve()
    results_dir = (results_dir or root / "results").resolve()
    claims_path = results_dir / "claims_status.json"
    payload = json.loads(claims_path.read_text(encoding="utf-8")) if claims_path.exists() else {}
    return audit_claim_evidence_payload(payload, root=root)


def claim_evidence_quality_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Claim Evidence Quality Report",
        "",
        f"- Verified: {payload.get('verified')}",
        f"- Claims audited: {payload.get('n_claims')}",
        f"- Source-mapped claims: {payload.get('n_source_mapped_claims')}",
        f"- Source links: {payload.get('n_source_links')}",
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
        lines.append("Every current claim has mapped source artifacts, structured evidence, no placeholder evidence literals, and valid referenced sources.")
    lines.append("")
    return "\n".join(lines)
