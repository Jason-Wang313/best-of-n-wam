from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from wam_inference_value.evaluation import ensure_result_dirs, results_dir, write_json


ROBOCASA_ARTIFACTS = [
    "benchmark_robocasa_smoke.json",
    "benchmark_robocasa_learned_wam.json",
    "benchmark_robocasa_multitask_wam.json",
    "benchmark_robocasa_broad_wam.json",
    "benchmark_robocasa_family12_wam.json",
    "benchmark_robocasa_family24_wam.json",
    "benchmark_robocasa_extra4_wam.json",
    "benchmark_robocasa_family28_wam.json",
    "benchmark_robocasa_family32_wam.json",
]

ROBOCASA_MICRO_ARTIFACTS = [
    "benchmark_robocasa_micro_rollout_extra.json",
]


def _load_result(name: str) -> dict[str, Any]:
    path = results_dir() / name
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _artifact_env_ids(payload: dict[str, Any]) -> list[str]:
    if not payload or not payload.get("verified", False):
        return []
    if payload.get("env_id"):
        return [str(payload["env_id"])]
    return [str(e) for e in payload.get("env_ids") or []]


def _micro_env_ids(payload: dict[str, Any]) -> list[str]:
    if not payload or not payload.get("verified", False):
        return []
    return [str(e) for e in payload.get("nondegenerate_env_ids") or []]


def _category(env_id: str) -> str:
    name = env_id.split("/", 1)[-1]
    if name.startswith("PickPlace"):
        return "pick_place"
    if name.startswith("Open"):
        return "open"
    if name.startswith("Close"):
        return "close"
    if name.startswith("Turn"):
        return "turn"
    if name.startswith("Move"):
        return "move"
    if name.startswith("Manipulate"):
        return "manipulate"
    if any(token in name for token in ("Clean", "Clear", "Wash", "Rinse", "Dry")):
        return "cleaning"
    if any(token in name for token in ("Cook", "Heat", "Boil", "Bake", "Oven", "Microwave", "Toast")):
        return "cooking"
    return "long_horizon_or_compositional"


def run() -> dict[str, Any]:
    ensure_result_dirs()
    report_dir = ROOT / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    try:
        import gymnasium as gym
        import robocasa  # noqa: F401 - registers environments
    except Exception as exc:
        summary = {
            "experiment": "benchmark_robocasa_catalog_probe",
            "attempted": True,
            "available": False,
            "verified": False,
            "reason": f"{type(exc).__name__}: {exc}",
        }
        write_json(results_dir() / "benchmark_robocasa_catalog_probe.json", summary)
        (report_dir / "robocasa_catalog_probe_report.md").write_text(
            "# RoboCasa Catalog Probe\n\n"
            "- status: `unavailable`\n"
            f"- reason: `{summary['reason']}`\n",
            encoding="utf-8",
        )
        return summary

    registry_ids = sorted(spec.id for spec in gym.envs.registry.values() if spec.id.startswith("robocasa/"))
    covered: set[str] = set()
    micro_covered: set[str] = set()
    artifact_rows: list[dict[str, Any]] = []
    for name in ROBOCASA_ARTIFACTS:
        payload = _load_result(name)
        env_ids = _artifact_env_ids(payload)
        covered.update(env_ids)
        artifact_rows.append(
            {
                "artifact": name,
                "present": bool(payload),
                "verified": bool(payload.get("verified", False)),
                "n_env_ids": len(env_ids),
                "env_ids": ";".join(env_ids),
                "evidence_level": "rollout_pool_learned_wam_or_smoke",
            }
        )
    for name in ROBOCASA_MICRO_ARTIFACTS:
        payload = _load_result(name)
        env_ids = _micro_env_ids(payload)
        micro_covered.update(env_ids)
        artifact_rows.append(
            {
                "artifact": name,
                "present": bool(payload),
                "verified": bool(payload.get("verified", False)),
                "n_env_ids": len(env_ids),
                "env_ids": ";".join(env_ids),
                "evidence_level": "micro_rollout_viability",
            }
        )

    registry_rows = []
    any_artifact = covered | micro_covered
    for env_id in registry_ids:
        registry_rows.append(
            {
                "env_id": env_id,
                "category": _category(env_id),
                "covered_by_verified_rollout_pool_artifact": env_id in covered,
                "covered_by_micro_rollout_probe": env_id in micro_covered,
                "covered_by_any_committed_artifact": env_id in any_artifact,
            }
        )
    registry_df = pd.DataFrame(registry_rows)
    artifact_df = pd.DataFrame(artifact_rows)
    registry_path = results_dir() / "tables" / "benchmark_robocasa_catalog_registry.csv"
    artifact_path = results_dir() / "tables" / "benchmark_robocasa_catalog_artifact_coverage.csv"
    registry_df.to_csv(registry_path, index=False)
    artifact_df.to_csv(artifact_path, index=False)

    category_counts = (
        registry_df.groupby("category")
        .agg(
            registered=("env_id", "count"),
            rollout_pool_covered=("covered_by_verified_rollout_pool_artifact", "sum"),
            micro_rollout_covered=("covered_by_micro_rollout_probe", "sum"),
            any_artifact_covered=("covered_by_any_committed_artifact", "sum"),
        )
        .reset_index()
        .to_dict(orient="records")
    )
    verified_artifact_env_ids = sorted(covered)
    micro_env_ids = sorted(micro_covered)
    any_artifact_env_ids = sorted(any_artifact)
    registry_count = len(registry_ids)
    covered_count = len(verified_artifact_env_ids)
    micro_count = len(micro_env_ids)
    any_count = len(any_artifact_env_ids)
    coverage_fraction = float(covered_count / registry_count) if registry_count else 0.0
    micro_coverage_fraction = float(micro_count / registry_count) if registry_count else 0.0
    any_coverage_fraction = float(any_count / registry_count) if registry_count else 0.0
    summary = {
        "experiment": "benchmark_robocasa_catalog_probe",
        "attempted": True,
        "available": True,
        "verified": registry_count >= 1,
        "registry_count": registry_count,
        "verified_artifact_task_count": covered_count,
        "coverage_fraction": coverage_fraction,
        "micro_rollout_task_count": micro_count,
        "micro_rollout_coverage_fraction": micro_coverage_fraction,
        "any_artifact_task_count": any_count,
        "any_artifact_coverage_fraction": any_coverage_fraction,
        "verified_artifact_env_ids": verified_artifact_env_ids,
        "micro_rollout_env_ids": micro_env_ids,
        "any_artifact_env_ids": any_artifact_env_ids,
        "category_counts": category_counts,
        "registry_path": str(registry_path),
        "artifact_coverage_path": str(artifact_path),
        "note": "Registry-level coverage audit only; rollout-pool coverage and micro-rollout viability are separate evidence tiers and neither validates uncovered tasks.",
    }
    write_json(results_dir() / "benchmark_robocasa_catalog_probe.json", summary)

    lines = [
        "# RoboCasa Catalog Probe",
        "",
        "- status: `verified`",
        f"- registered RoboCasa task IDs: `{registry_count}`",
        f"- task IDs covered by verified rollout-pool artifacts: `{covered_count}`",
        f"- rollout-pool coverage fraction: `{coverage_fraction:.4f}`",
        f"- task IDs covered by micro-rollout viability probes: `{micro_count}`",
        f"- any-artifact task coverage: `{any_count}`",
        f"- any-artifact coverage fraction: `{any_coverage_fraction:.4f}`",
        "",
        "## Coverage By Category",
        "",
    ]
    for row in category_counts:
        lines.append(
            f"- `{row['category']}`: rollout-pool `{int(row['rollout_pool_covered'])}`, "
            f"micro-rollout `{int(row['micro_rollout_covered'])}`, any `{int(row['any_artifact_covered'])}` / registered `{int(row['registered'])}`"
        )
    lines.extend(
        [
            "",
            "This is a registry coverage audit. It deliberately does not promote uncovered task IDs to benchmark evidence; uncovered IDs still need reset, rollout-pool, learned-WAM, and CI artifacts before the README can claim them.",
        ]
    )
    (report_dir / "robocasa_catalog_probe_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(run())
