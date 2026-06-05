from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .evaluation import results_dir, write_json


VALID_RESOLUTION_CLASSES = {
    "external_physical_evidence_required",
    "runtime_policy_eval_required",
    "benchmark_coverage_required",
    "local_runtime_dependency_required",
    "mathematical_scope_boundary",
}


BLOCKER_REQUIREMENTS: dict[str, dict[str, Any]] = {
    "real_robot_hil": {
        "blocker_class": "hardware_or_hil_trial_absent",
        "resolution_class": "external_physical_evidence_required",
        "local_progress_status": "No local code or simulator artifact can promote this endpoint; it needs physical or HIL trial metrics.",
        "required_artifacts": [
            "results/real_robot_hil_probe.json",
            "reports/real_robot_hil_probe_report.md",
        ],
    },
    "modern_vla_libero": {
        "blocker_class": "modern_vla_execution_or_eval_absent",
        "resolution_class": "runtime_policy_eval_required",
        "local_progress_status": "Runtime access exists, but the promoted endpoint needs heldout sparse-success modern-VLA episodes with nonzero successes and CIs.",
        "required_artifacts": [
            "results/modern_vla_availability_probe.json",
            "reports/modern_vla_availability_probe_report.md",
            "results/modern_vla_libero_execution_probe.json",
            "reports/modern_vla_libero_execution_probe_report.md",
            "results/modern_vla_libero_policy_eval.json",
            "reports/modern_vla_libero_policy_eval_report.md",
            "results/benchmark_libero_visual_language_bc_policy.json",
        ],
    },
    "full_robocasa_wide": {
        "blocker_class": "registry_coverage_incomplete",
        "resolution_class": "benchmark_coverage_required",
        "local_progress_status": "Further local RoboCasa sweeps can improve coverage, but full-suite promotion requires coverage of every declared registry task or a narrower claim.",
        "required_artifacts": [
            "results/benchmark_robocasa_catalog_probe.json",
            "results/benchmark_robocasa_residual_triage.json",
            "reports/robocasa_residual_triage_report.md",
        ],
    },
    "maniskill_visual_ee": {
        "blocker_class": "renderer_or_ee_dependency_blocked",
        "resolution_class": "local_runtime_dependency_required",
        "local_progress_status": "State-mode evidence is complete; RGB/RGB-D and EE-control promotion needs a working Vulkan/SAPIEN renderer and robotics Pinocchio API.",
        "required_artifacts": [
            "results/benchmark_maniskill_visual_probe.json",
            "results/benchmark_maniskill_dependency_probe.json",
            "reports/maniskill_visual_blocker_report.md",
            "reports/maniskill_dependency_blocker_report.md",
        ],
    },
    "universal_wam_training_recipe": {
        "blocker_class": "unrestricted_universal_recipe_no_free_lunch",
        "resolution_class": "mathematical_scope_boundary",
        "local_progress_status": "This unrestricted universal-recipe endpoint is intentionally future-only; current artifacts support scoped optimization, not a universal proof.",
        "required_artifacts": [
            "results/universal_wam_train_inference_optimizer.json",
            "results/universal_recipe_boundary.json",
            "reports/universal_recipe_boundary_report.md",
        ],
    },
}


@dataclass(frozen=True)
class AuditCheck:
    name: str
    ok: bool
    detail: str


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _readiness_rows(readiness: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = readiness.get("rows") if isinstance(readiness.get("rows"), list) else []
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        if isinstance(row, dict) and row.get("frontier_id"):
            out[str(row["frontier_id"])] = row
    return out


def _artifact(root: Path, relative: str) -> dict[str, Any]:
    path = root / relative
    return {
        "path": relative,
        "exists": path.exists(),
        "bytes": path.stat().st_size if path.exists() and path.is_file() else 0,
    }


def _frontier_evidence(root: Path, frontier_id: str) -> dict[str, Any]:
    results = root / "results"
    if frontier_id == "real_robot_hil":
        probe = _load_json(results / "real_robot_hil_probe.json")
        return {
            "probe_verified": probe.get("verified"),
            "possible_hardware_device_count": probe.get("possible_hardware_device_count"),
            "trial_metric_artifact_count": probe.get("trial_metric_artifact_count"),
            "claim_ready": probe.get("real_robot_or_hil_claim_ready"),
        }
    if frontier_id == "modern_vla_libero":
        probe = _load_json(results / "modern_vla_availability_probe.json")
        execution = _load_json(results / "modern_vla_libero_execution_probe.json")
        policy_eval = _load_json(results / "modern_vla_libero_policy_eval.json")
        last_attempt = _load_json(results / "modern_vla_libero_policy_eval_last_attempt.json")
        attempt_history = policy_eval.get("attempt_history")
        if not isinstance(attempt_history, list):
            attempt_history = last_attempt.get("attempt_history")
        if not isinstance(attempt_history, list):
            attempt_history = []
        return {
            "probe_verified": probe.get("verified"),
            "vla_package_importable": probe.get("vla_package_importable"),
            "local_vla_like_count": probe.get("local_vla_like_count"),
            "hf_reachable_count": probe.get("hf_reachable_count"),
            "vla_libero_joint_runtime_available": probe.get("vla_libero_joint_runtime_available"),
            "pretrained_vla_loaded": probe.get("pretrained_vla_loaded"),
            "pretrained_vla_parameter_count": probe.get("pretrained_vla_parameter_count"),
            "execution_attempted": execution.get("attempted"),
            "execution_verified": execution.get("verified"),
            "execution_failure_stage": execution.get("failure_stage"),
            "execution_error_type": execution.get("error_type"),
            "ready_for_policy_eval": probe.get("ready_for_policy_eval"),
            "policy_eval_verified": policy_eval.get("verified"),
            "policy_eval_episodes": policy_eval.get("eval_episodes"),
            "policy_eval_successes": policy_eval.get("eval_successes"),
            "policy_eval_success_ci": policy_eval.get("success_ci"),
            "last_attempt_recorded": bool(last_attempt),
            "last_attempt_verified": last_attempt.get("verified"),
            "last_attempt_horizon": last_attempt.get("horizon"),
            "last_attempt_max_steps": last_attempt.get("max_steps"),
            "last_attempt_requested_eval_seeds": last_attempt.get("requested_eval_seeds"),
            "last_attempt_failure_stage": last_attempt.get("failure_stage"),
            "last_attempt_error_type": last_attempt.get("error_type"),
            "last_attempt_child_returncode": last_attempt.get("child_returncode"),
            "attempt_history_count": len(attempt_history),
            "attempt_history_error_types": sorted(
                {
                    str(item.get("error_type"))
                    for item in attempt_history
                    if isinstance(item, dict) and item.get("error_type")
                }
            ),
        }
    if frontier_id == "full_robocasa_wide":
        catalog = _load_json(results / "benchmark_robocasa_catalog_probe.json")
        triage = _load_json(results / "benchmark_robocasa_residual_triage.json")
        return {
            "catalog_verified": catalog.get("verified"),
            "registry_count": catalog.get("registry_count"),
            "any_artifact_task_count": catalog.get("any_artifact_task_count"),
            "any_artifact_coverage_fraction": catalog.get("any_artifact_coverage_fraction"),
            "triage_verified": triage.get("verified"),
            "unattempted": triage.get("unattempted"),
            "attempted_not_covered": triage.get("attempted_not_covered"),
        }
    if frontier_id == "maniskill_visual_ee":
        visual = _load_json(results / "benchmark_maniskill_visual_probe.json")
        deps = _load_json(results / "benchmark_maniskill_dependency_probe.json")
        return {
            "visual_attempted": visual.get("attempted"),
            "any_visual_success": visual.get("any_visual_success"),
            "visual_attempt_count": visual.get("visual_attempt_count"),
            "any_ee_control_success": visual.get("any_ee_control_success"),
            "pinocchio_api_available": deps.get("pinocchio_api_available"),
            "pinocchio_missing_symbols": deps.get("pinocchio_missing_symbols"),
            "external_env_python_count": deps.get("external_env_python_count"),
            "external_env_pinocchio_api_any_available": deps.get("external_env_pinocchio_api_any_available"),
            "external_env_pinocchio_api_available_pythons": deps.get("external_env_pinocchio_api_available_pythons"),
        }
    if frontier_id == "universal_wam_training_recipe":
        optimizer = _load_json(results / "universal_wam_train_inference_optimizer.json")
        boundary = _load_json(results / "universal_recipe_boundary.json")
        return {
            "optimizer_verified": optimizer.get("verified"),
            "not_a_universal_proof": optimizer.get("not_a_universal_proof"),
            "boundary_verified": boundary.get("verified"),
            "boundary_type": boundary.get("result_type"),
            "boundary_regret_lb": boundary.get("randomized_worst_case_regret_lower_bound"),
        }
    return {}


def _add(checks: list[AuditCheck], name: str, ok: bool, detail: str) -> None:
    checks.append(AuditCheck(name=name, ok=bool(ok), detail=detail))


def build_ideal_frontier_blocker_audit(root: Path, output_results_dir: Path | None = None) -> dict[str, Any]:
    root = root.resolve()
    output_results_dir = (output_results_dir or results_dir()).resolve()
    readiness = _load_json(output_results_dir / "ideal_frontier_readiness.json")
    rows = _readiness_rows(readiness)
    checks: list[AuditCheck] = []
    blocker_rows: list[dict[str, Any]] = []

    _add(checks, "readiness_artifact_verified", readiness.get("verified") is True, f"verified={readiness.get('verified')}")
    _add(
        checks,
        "no_ideal_frontier_promoted",
        int(readiness.get("n_ready_to_promote") or 0) == 0,
        f"ready={readiness.get('n_ready_to_promote')}/{readiness.get('n_frontiers')}",
    )

    for frontier_id, spec in BLOCKER_REQUIREMENTS.items():
        row = rows.get(frontier_id, {})
        missing_signals = row.get("missing_signals") if isinstance(row.get("missing_signals"), list) else []
        signals = row.get("signals") if isinstance(row.get("signals"), list) else []
        missing_details = [
            str(signal.get("detail") or "")
            for signal in signals
            if isinstance(signal, dict) and signal.get("name") in missing_signals
        ]
        artifacts = [_artifact(root, relative) for relative in spec["required_artifacts"]]
        missing_artifacts = [artifact for artifact in artifacts if not artifact["exists"] or int(artifact["bytes"]) <= 0]
        next_action = str(row.get("next_action") or "")
        evidence = _frontier_evidence(root, frontier_id)
        blocker_rows.append(
            {
                "frontier_id": frontier_id,
                "blocker_class": spec["blocker_class"],
                "resolution_class": spec["resolution_class"],
                "local_progress_status": spec["local_progress_status"],
                "ready_to_promote": bool(row.get("ready_to_promote")),
                "missing_signals": missing_signals,
                "next_action": next_action,
                "required_artifacts": artifacts,
                "evidence": evidence,
            }
        )
        _add(checks, f"{frontier_id}_row_present", bool(row), f"present={bool(row)}")
        _add(checks, f"{frontier_id}_still_unpromoted", row.get("ready_to_promote") is False, f"ready={row.get('ready_to_promote')}")
        _add(checks, f"{frontier_id}_has_missing_signals", bool(missing_signals), f"missing={missing_signals}")
        _add(
            checks,
            f"{frontier_id}_resolution_class_valid",
            str(spec.get("resolution_class")) in VALID_RESOLUTION_CLASSES,
            f"resolution_class={spec.get('resolution_class')}",
        )
        _add(
            checks,
            f"{frontier_id}_local_progress_status_present",
            len(str(spec.get("local_progress_status") or "")) >= 48,
            f"local_progress_status={spec.get('local_progress_status')}",
        )
        _add(
            checks,
            f"{frontier_id}_missing_signal_details_present",
            len(missing_details) == len(missing_signals) and all(len(detail) >= 12 for detail in missing_details),
            f"detail_count={len(missing_details)}, missing_count={len(missing_signals)}",
        )
        _add(checks, f"{frontier_id}_next_action_present", len(next_action) >= 24, f"next_action={next_action}")
        _add(checks, f"{frontier_id}_blocker_artifacts_present", not missing_artifacts, f"missing_artifacts={missing_artifacts}")

    issues = [check for check in checks if not check.ok]
    payload = {
        "experiment": "ideal_frontier_blocker_audit",
        "verified": len(issues) == 0,
        "scope": "documents why ideal-frontier claims remain unpromoted; not evidence for those claims",
        "n_frontiers": len(BLOCKER_REQUIREMENTS),
        "n_ready_to_promote": int(readiness.get("n_ready_to_promote") or 0),
        "blocker_rows": blocker_rows,
        "n_checks": len(checks),
        "n_issues": len(issues),
        "checks": [check.__dict__ for check in checks],
        "issues": [check.__dict__ for check in issues],
    }
    write_json(output_results_dir / "ideal_frontier_blocker_audit.json", payload)
    return payload


def ideal_frontier_blocker_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Ideal Frontier Blocker Audit",
        "",
        f"- verified: `{payload.get('verified')}`",
        f"- scope: `{payload.get('scope')}`",
        f"- ready to promote: `{payload.get('n_ready_to_promote')}`",
        f"- issues: `{payload.get('n_issues')}`",
        "",
        "This report is blocker evidence, not validation evidence for the ideal endpoints.",
        "",
    ]
    for row in payload.get("blocker_rows") or []:
        lines.extend(
            [
                f"## {row.get('frontier_id')}",
                "",
                f"- blocker class: `{row.get('blocker_class')}`",
                f"- resolution class: `{row.get('resolution_class')}`",
                f"- local progress status: {row.get('local_progress_status')}",
                f"- ready to promote: `{row.get('ready_to_promote')}`",
                f"- missing signals: `{row.get('missing_signals')}`",
                f"- next action: {row.get('next_action')}",
                "",
                "### Evidence",
                "",
            ]
        )
        evidence = row.get("evidence") if isinstance(row.get("evidence"), dict) else {}
        for key, value in evidence.items():
            lines.append(f"- `{key}`: `{value}`")
        lines.append("")
    return "\n".join(lines)
