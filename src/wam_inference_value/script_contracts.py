from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


CORE_GATE_SEQUENCE = [
    "scripts/artifact_integrity.py --fail-on-error",
    "scripts/result_consistency.py --fail-on-error",
    "scripts/raw_result_recompute.py --fail-on-error",
    "scripts/table_schema.py --fail-on-error",
    "scripts/artifact_manifest.py --fail-on-error",
    "scripts/figure_quality.py --fail-on-error",
    "scripts/narrative_consistency.py --fail-on-error",
    "scripts/script_contracts.py --fail-on-error",
    "scripts/claims_status.py",
    "scripts/claim_semantics.py --fail-on-error",
    "scripts/claims_status.py",
    "scripts/claim_evidence_quality.py --fail-on-error",
    "scripts/claims_status.py",
    "scripts/claim_ledger_integrity.py --fail-on-error",
    "scripts/claims_status.py",
    "scripts/claim_semantics.py --fail-on-error",
    "scripts/claims_status.py",
    "scripts/claim_evidence_quality.py --fail-on-error",
    "scripts/claims_status.py",
    "scripts/claim_ledger_integrity.py --fail-on-error",
]
CORE_SCRIPT_REQUIREMENTS = {
    "scripts/run_smoke.sh": [
        "-m pytest -q",
        "experiments/train_learned_wam_lite.py",
        "experiments/learned_wam_vs_analytic_wam.py",
        "experiments/exact_rollout_law_validation.py",
        "experiments/nonstationary_dynamics_extension.py",
    ],
    "scripts/run_learned_wam_toy.sh": [
        "experiments/train_learned_wam_lite.py",
        "experiments/exact_rollout_law_validation.py",
        "experiments/score_function_comparison.py",
        "experiments/real_vs_imagined_utility_gap.py",
        "experiments/adaptive_rollout_allocation.py",
        "experiments/closed_loop_receding_horizon_eval.py",
        "experiments/learned_wam_vs_analytic_wam.py",
    ],
    "scripts/run_all.sh": [
        "-m pytest -q",
        "experiments/exact_rollout_law_validation.py",
        "experiments/auc_vs_moment_hierarchy.py",
        "experiments/pilot_to_heldout_prediction.py",
        "experiments/score_function_comparison.py",
        "experiments/real_vs_imagined_utility_gap.py",
        "experiments/adaptive_rollout_allocation.py",
        "experiments/closed_loop_receding_horizon_eval.py",
        "experiments/nonstationary_dynamics_extension.py",
    ],
    "scripts/run_inference_audit.sh": [
        "experiments/inference_audit_framework.py",
        "experiments/scorer_repair_experiment.py",
        "experiments/imagination_scaling_law.py",
    ],
}
MAXOUT_REQUIRED_SNIPPETS = [
    "-m pytest -q",
    "bash scripts/run_smoke.sh",
    "bash scripts/run_learned_wam_toy.sh",
    "bash scripts/run_multi_env.sh",
    "bash scripts/run_benchmark_full.sh",
    "bash scripts/run_visual_optional.sh",
    "bash scripts/run_inference_audit.sh",
    "scripts/write_maxout_reports.py",
]
OPTIONAL_SCRIPT_REQUIREMENTS = {
    "scripts/run_benchmark_full.sh": {
        "snippets": [
            "bash scripts/run_benchmark_smoke.sh",
            "experiments/benchmark_gym_manip_suite.py",
            "experiments/benchmark_gym_robotics_suite.py",
            "experiments/benchmark_metaworld_suite.py",
            "experiments/benchmark_robosuite_suite.py",
            "experiments/benchmark_maniskill_suite.py",
            "bash scripts/run_benchmark_visual_optional.sh",
        ],
        "guards": ["ROBOCASA_PYTHON", "LIBERO_PYTHON"],
    },
    "scripts/run_visual_optional.sh": {
        "snippets": ["experiments/visual_optional.py", "bash scripts/run_benchmark_visual_optional.sh"],
        "guards": [],
    },
}


@dataclass(frozen=True)
class ScriptContractCheck:
    name: str
    ok: bool
    detail: str


def add(checks: list[ScriptContractCheck], name: str, ok: bool, detail: str) -> None:
    checks.append(ScriptContractCheck(name=name, ok=bool(ok), detail=detail))


def read_text(root: Path, script: str) -> str:
    path = root / script
    return path.read_text(encoding="utf-8") if path.exists() else ""


def ordered_subsequence(text: str, snippets: list[str]) -> bool:
    index = 0
    for snippet in snippets:
        found = text.find(snippet, index)
        if found < 0:
            return False
        index = found + len(snippet)
    return True


def count_occurrences(text: str, snippet: str) -> int:
    return text.count(snippet)


def audit_core_script(root: Path, script: str, required_snippets: list[str], checks: list[ScriptContractCheck]) -> None:
    path = root / script
    text = read_text(root, script)
    label = Path(script).stem
    add(checks, f"{label}_exists", path.exists(), f"path={script}")
    add(checks, f"{label}_strict_bash", "set -euo pipefail" in text, "requires set -euo pipefail")
    add(checks, f"{label}_python_fallback", "python.exe" in text and "python3" in text, "python fallback ladder present")
    missing = [snippet for snippet in required_snippets if snippet not in text]
    add(checks, f"{label}_required_steps", not missing, f"missing={missing}")
    add(checks, f"{label}_gate_sequence", ordered_subsequence(text, CORE_GATE_SEQUENCE), "core gate sequence is ordered")
    add(checks, f"{label}_raw_recompute_gate", count_occurrences(text, "scripts/raw_result_recompute.py --fail-on-error") >= 1, "raw recompute gate runs after result consistency")
    add(checks, f"{label}_table_schema_gate", count_occurrences(text, "scripts/table_schema.py --fail-on-error") >= 1, "table schema gate runs after raw recompute")
    add(checks, f"{label}_artifact_manifest_gate", count_occurrences(text, "scripts/artifact_manifest.py --fail-on-error") >= 1, "artifact manifest gate runs after table schema")
    add(checks, f"{label}_figure_quality_gate", count_occurrences(text, "scripts/figure_quality.py --fail-on-error") >= 1, "figure quality gate runs after artifact manifest")
    add(checks, f"{label}_double_claim_semantics", count_occurrences(text, "scripts/claim_semantics.py --fail-on-error") >= 2, "claim semantic gate runs before both ledger gates")
    add(checks, f"{label}_double_claim_evidence_quality", count_occurrences(text, "scripts/claim_evidence_quality.py --fail-on-error") >= 2, "claim evidence gate runs before both ledger gates")
    add(checks, f"{label}_double_claim_ledger", count_occurrences(text, "scripts/claim_ledger_integrity.py --fail-on-error") >= 2, "ledger gate runs after both claim-status writes")


def audit_maxout(root: Path, checks: list[ScriptContractCheck]) -> None:
    script = "scripts/run_maxout_all.sh"
    text = read_text(root, script)
    path = root / script
    add(checks, "run_maxout_all_exists", path.exists(), f"path={script}")
    add(checks, "run_maxout_all_strict_bash", "set -euo pipefail" in text, "requires set -euo pipefail")
    missing = [snippet for snippet in MAXOUT_REQUIRED_SNIPPETS if snippet not in text]
    add(checks, "run_maxout_all_required_steps", not missing, f"missing={missing}")
    add(checks, "run_maxout_all_pre_report_gates", count_occurrences(text, "scripts/artifact_integrity.py --fail-on-error") >= 2, "artifact gate before and after reports")
    add(checks, "run_maxout_all_post_report_sequence", "scripts/write_maxout_reports.py" in text and ordered_subsequence(text, ["scripts/write_maxout_reports.py"] + CORE_GATE_SEQUENCE), "post-report gate sequence is ordered")


def audit_optional_script(root: Path, script: str, requirements: dict[str, list[str]], checks: list[ScriptContractCheck]) -> None:
    path = root / script
    text = read_text(root, script)
    label = Path(script).stem
    add(checks, f"{label}_exists", path.exists(), f"path={script}")
    add(checks, f"{label}_strict_bash", "set -euo pipefail" in text, "requires set -euo pipefail")
    missing = [snippet for snippet in requirements.get("snippets", []) if snippet not in text]
    add(checks, f"{label}_required_steps", not missing, f"missing={missing}")
    missing_guards = [guard for guard in requirements.get("guards", []) if f'${{{guard}:-}}' not in text and f'"${guard}"' not in text]
    add(checks, f"{label}_optional_guards", not missing_guards, f"missing_guards={missing_guards}")


def audit_script_contracts(root: Path) -> dict[str, Any]:
    root = root.resolve()
    checks: list[ScriptContractCheck] = []
    for script, required in CORE_SCRIPT_REQUIREMENTS.items():
        audit_core_script(root, script, required, checks)
    audit_maxout(root, checks)
    for script, requirements in OPTIONAL_SCRIPT_REQUIREMENTS.items():
        audit_optional_script(root, script, requirements, checks)

    issues = [check for check in checks if not check.ok]
    return {
        "experiment": "script_contracts",
        "verified": len(issues) == 0,
        "n_scripts": len(CORE_SCRIPT_REQUIREMENTS) + len(OPTIONAL_SCRIPT_REQUIREMENTS) + 1,
        "n_checks": len(checks),
        "n_issues": len(issues),
        "core_gate_sequence": CORE_GATE_SEQUENCE,
        "checks": [check.__dict__ for check in checks],
        "issues": [check.__dict__ for check in issues],
    }


def script_contracts_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Script Contract Report",
        "",
        f"- Verified: {payload.get('verified')}",
        f"- Scripts audited: {payload.get('n_scripts')}",
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
        lines.append("Canonical scripts preserve required experiment steps, strict Bash mode, optional benchmark guards, and ordered artifact/result/raw-recompute/table-schema/manifest/figure-quality/narrative/script-contract/claim/semantic/evidence/ledger gates.")
    lines.append("")
    return "\n".join(lines)
