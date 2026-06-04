from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ReadinessSignal:
    name: str
    ok: bool
    detail: str


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _model_exists(root: Path, payload: dict[str, Any]) -> bool:
    model_path = payload.get("model_path")
    if not model_path:
        return False
    path = Path(str(model_path))
    if not path.is_absolute():
        path = root / path
    return path.exists() and path.stat().st_size > 0


def _signal(signals: list[ReadinessSignal], name: str, ok: bool, detail: str) -> None:
    signals.append(ReadinessSignal(name=name, ok=bool(ok), detail=detail))


def _row(frontier_id: str, signals: list[ReadinessSignal], *, next_action: str) -> dict[str, Any]:
    missing = [signal.name for signal in signals if not signal.ok]
    return {
        "frontier_id": frontier_id,
        "ready_to_promote": not missing,
        "n_signals": len(signals),
        "n_met_signals": len(signals) - len(missing),
        "n_missing_signals": len(missing),
        "missing_signals": missing,
        "signals": [signal.__dict__ for signal in signals],
        "next_action": next_action,
    }


def audit_ideal_frontier_readiness(root: Path, results_dir: Path | None = None) -> dict[str, Any]:
    root = root.resolve()
    results_dir = (results_dir or root / "results").resolve()

    libero_vl = _load_json(results_dir / "benchmark_libero_visual_language_bc_policy.json")
    robocasa_catalog = _load_json(results_dir / "benchmark_robocasa_catalog_probe.json")
    maniskill_visual = _load_json(results_dir / "benchmark_maniskill_visual_probe.json")
    maniskill_deps = _load_json(results_dir / "benchmark_maniskill_dependency_probe.json")
    publication_scope = _load_json(results_dir / "publication_scope.json")

    rows: list[dict[str, Any]] = []

    real_robot: list[ReadinessSignal] = []
    real_artifacts = [
        path
        for pattern in ("*real_robot*", "*hardware*", "*hil*")
        for path in results_dir.glob(pattern)
        if path.is_file()
    ]
    _signal(real_robot, "real_robot_or_hil_artifact_present", bool(real_artifacts), f"files={len(real_artifacts)}")
    _signal(real_robot, "real_world_success_metrics_present", False, "no physical trial metric artifact is declared")
    rows.append(
        _row(
            "real_robot_hil",
            real_robot,
            next_action="Collect real robot or hardware-in-the-loop rollout/control logs with trial IDs and success or utility metrics.",
        )
    )

    policy = libero_vl.get("policy") if isinstance(libero_vl.get("policy"), dict) else {}
    ci = (libero_vl.get("confidence_intervals") or {}).get("eval_success_rate") or {}
    policy_type = str(policy.get("type") or "").lower()
    no_shortcuts = (
        policy.get("uses_rgb") is True
        and policy.get("uses_language") is True
        and policy.get("uses_simulator_object_state") is False
        and policy.get("uses_task_id") is False
        and policy.get("uses_phase_index") is False
        and policy.get("uses_target_point_command") is False
    )
    modern_vla: list[ReadinessSignal] = []
    _signal(
        modern_vla,
        "libero_rgb_language_policy_evaluated",
        libero_vl.get("verified") is True and int(libero_vl.get("eval_episodes") or 0) > 0,
        f"verified={libero_vl.get('verified')}, eval_episodes={libero_vl.get('eval_episodes')}",
    )
    _signal(modern_vla, "no_shortcut_eval_interface", no_shortcuts, f"policy={policy}")
    _signal(modern_vla, "sparse_success_ci_reported", int(ci.get("n") or 0) > 0, f"ci={ci}")
    _signal(modern_vla, "model_artifact_present", _model_exists(root, libero_vl), f"model_path={libero_vl.get('model_path')}")
    _signal(
        modern_vla,
        "modern_vla_model_class",
        bool(policy_type) and "knn" not in policy_type and "scripted" not in policy_type,
        f"policy_type={policy_type!r}",
    )
    _signal(
        modern_vla,
        "current_runtime_can_rerun_libero",
        False,
        "current interpreter cannot import LIBERO unless LIBERO_PYTHON/LIBERO_SOURCE_PATH are supplied",
    )
    rows.append(
        _row(
            "modern_vla_libero",
            modern_vla,
            next_action="Run a neural RGB/proprio/language policy under a compatible LIBERO runtime and report heldout sparse-success CIs.",
        )
    )

    registry_count = int(robocasa_catalog.get("registry_count") or 0)
    rollout_count = int(robocasa_catalog.get("verified_artifact_task_count") or 0)
    any_count = int(robocasa_catalog.get("any_artifact_task_count") or 0)
    category_counts = robocasa_catalog.get("category_counts") if isinstance(robocasa_catalog.get("category_counts"), list) else []
    missing_categories = [
        str(row.get("category"))
        for row in category_counts
        if int(row.get("any_artifact_covered") or 0) < int(row.get("registered") or 0)
    ]
    robocasa: list[ReadinessSignal] = []
    _signal(robocasa, "catalog_probe_verified", robocasa_catalog.get("verified") is True, f"verified={robocasa_catalog.get('verified')}")
    _signal(
        robocasa,
        "full_registry_rollout_pool_coverage",
        registry_count > 0 and rollout_count == registry_count,
        f"rollout_pool_covered={rollout_count}/{registry_count}",
    )
    _signal(
        robocasa,
        "full_registry_any_artifact_coverage",
        registry_count > 0 and any_count == registry_count,
        f"any_artifact_covered={any_count}/{registry_count}",
    )
    _signal(robocasa, "all_categories_fully_covered", not missing_categories and bool(category_counts), f"missing={missing_categories}")
    rows.append(
        _row(
            "full_robocasa_wide",
            robocasa,
            next_action="Extend RoboCasa artifacts from sampled/stratified coverage to every declared registry task or state a narrower benchmark scope.",
        )
    )

    maniskill: list[ReadinessSignal] = []
    _signal(maniskill, "visual_probe_attempted", bool(maniskill_visual.get("attempted")), f"attempted={maniskill_visual.get('attempted')}")
    _signal(
        maniskill,
        "rgb_or_rgbd_success",
        maniskill_visual.get("any_visual_success") is True,
        f"visual_success={maniskill_visual.get('any_visual_success')}, attempts={maniskill_visual.get('visual_attempt_count')}",
    )
    _signal(
        maniskill,
        "ee_control_success",
        maniskill_visual.get("any_ee_control_success") is True,
        f"ee_success={maniskill_visual.get('any_ee_control_success')}, attempts={maniskill_visual.get('ee_control_attempt_count')}",
    )
    _signal(
        maniskill,
        "pinocchio_available_for_ee",
        maniskill_deps.get("pinocchio_import_available") is True,
        f"pinocchio={maniskill_deps.get('pinocchio_import_available')}",
    )
    rows.append(
        _row(
            "maniskill_visual_ee",
            maniskill,
            next_action="Clear the Vulkan descriptor-pool renderer failure and install a compatible Pinocchio stack for EE controllers.",
        )
    )

    universal: list[ReadinessSignal] = []
    optimizer_files = [
        root / "src" / "wam_inference_value" / "train_inference_optimizer.py",
        root / "experiments" / "universal_wam_train_inference_optimizer.py",
    ]
    _signal(
        universal,
        "future_scope_guard_present",
        publication_scope.get("verified") is True,
        f"publication_scope_verified={publication_scope.get('verified')}",
    )
    _signal(universal, "optimizer_artifact_present", any(path.exists() for path in optimizer_files), f"checked={[str(path.relative_to(root)) for path in optimizer_files]}")
    _signal(universal, "cross_environment_optimizer_evidence_present", False, "no cross-environment optimizer result artifact is declared")
    rows.append(
        _row(
            "universal_wam_training_recipe",
            universal,
            next_action="Build and evaluate a train/inference optimizer across data scale, model class, horizon, scorer, safety, and rollout budget.",
        )
    )

    ready = [row for row in rows if row["ready_to_promote"]]
    return {
        "experiment": "ideal_frontier_readiness",
        "verified": True,
        "n_frontiers": len(rows),
        "n_ready_to_promote": len(ready),
        "n_not_ready_to_promote": len(rows) - len(ready),
        "ready_frontier_ids": [row["frontier_id"] for row in ready],
        "not_ready_frontier_ids": [row["frontier_id"] for row in rows if not row["ready_to_promote"]],
        "rows": rows,
        "note": "Readiness audit only. A frontier is not a supported result until ready_to_promote is true and claims_status promotes it with artifact-backed evidence.",
    }


def ideal_frontier_readiness_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Ideal Frontier Readiness",
        "",
        f"- Verified audit: `{payload.get('verified')}`",
        f"- Frontiers audited: `{payload.get('n_frontiers')}`",
        f"- Ready to promote: `{payload.get('n_ready_to_promote')}`",
        f"- Not ready to promote: `{payload.get('n_not_ready_to_promote')}`",
        "",
        "This report is a gap audit, not a result promotion mechanism.",
        "",
        "## Frontier Matrix",
        "",
    ]
    for row in payload.get("rows") or []:
        lines.append(f"- `{row.get('frontier_id')}`: ready_to_promote=`{row.get('ready_to_promote')}`; missing={row.get('missing_signals')}")
        for signal in row.get("signals") or []:
            lines.append(f"  - {signal.get('name')}: `{signal.get('ok')}` ({signal.get('detail')})")
        lines.append(f"  Next action: {row.get('next_action')}")
    lines.append("")
    return "\n".join(lines)
